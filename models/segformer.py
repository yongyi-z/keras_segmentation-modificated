import tensorflow as tf
import keras
from keras import layers
from keras.models import *
from keras.layers import *
from keras.models import Model
from .model_utils import get_segmentation_model
import numpy as np

#source: https://www.geeksforgeeks.org/python-flatten-nested-tuples/
def flatten_tuples(t):
    for x in t:
        if isinstance(x, tuple):
            yield from flatten_tuples(x)
        else:
            yield x


#from https://github.com/ACSEkevin/An-Overview-of-Segformer-and-Details-Description

def convolutional_block(inputs, n_filters, kernel_size, strides=1, norm=True, activation=None):
    x = Conv2D(n_filters, kernel_size, strides, padding='same')(inputs)
    x = BatchNormalization(momentum=.99, epsilon=1e-5)(x) if norm is True else x
    x = Activation(activation)(x) if activation else x
    return x


def decoder_mlp(inputs, embed_dim):
    """
    input shape -> [batches, height, width, embed_dim]
    :return:  shape -> [batches, n_patches, embed_dim]
    """
    batches, height, width, channels = inputs.shape
    x = tf.reshape(inputs, shape=[-1, height * width, channels])
    x = Dense(embed_dim, use_bias=True)(x)

    return x


def seg_former_decoder_block(inputs, embed_dim, up_size=(4, 4)):
    """
    inputs: shape -> [batches, height, width, embed_dim]
    :return: shape -> [batches, height, width, embed_dim]
    """
    batches, height, width, channels = inputs.shape
    x = decoder_mlp(inputs, embed_dim)
    x = tf.reshape(x, shape=[-1, height, width, embed_dim])
    x = UpSampling2D(size=up_size, interpolation='bilinear')(x)

    return x


def seg_former_head(features, embed_dim, n_classes, drop_rate=0.):
    assert len(features) == 4
    assert len(set(feature.shape for feature in features)) == 1

    x = Concatenate(axis=-1)(features)
    x = convolutional_block(x, n_filters=embed_dim, kernel_size=1, norm=True, activation='relu')
    x = Dropout(rate=drop_rate)(x)
    x = Conv2D(n_classes, kernel_size=1, activation='softmax')(x)
    return x

def overlap_patch_embedding(inputs, n_filters, kernel_size, strides):
    x = Conv2D(n_filters, kernel_size=kernel_size, strides=strides, padding='same')(inputs)
    print('overlap patch embedding, x.shape: {}'.format(x.shape))
    batches, height, width, embed_dim = x.shape
    x = tf.reshape(x, shape=[-1, height * width, embed_dim])

    return LayerNormalization()(x), height, width


def efficient_multi_head_attention(inputs, height, width, embed_dim, n_heads, scaler=None,
                                   use_bias=True, sr_ratio: int = 1,
                                   attention_drop_rate=0., projection_drop_rate=0.):
    batches, n_patches, channels = inputs.shape
    assert channels == embed_dim
    assert height and width and height * width == n_patches

    head_dim = embed_dim // n_heads
    scaler = head_dim ** -0.5 if scaler is None else scaler

    query = Dense(embed_dim, use_bias=use_bias)(inputs)
    query = tf.reshape(query, shape=[-1, n_patches, n_heads, head_dim])
    query = tf.transpose(query, perm=[0, 2, 1, 3])

    if sr_ratio > 1:
        inputs = tf.reshape(inputs, shape=[-1, height, width, embed_dim])
        # shape -> [batches, height/sr, width/sr, embed_dim]
        inputs = Conv2D(embed_dim, kernel_size=sr_ratio, strides=sr_ratio, padding='same')(inputs)
        inputs = LayerNormalization()(inputs)
        # shape -> [batches, height * width/sr ** 2, embed_dim]
        inputs = tf.reshape(inputs, shape=[-1, (height * width) // (sr_ratio ** 2), embed_dim])

    key_value = Dense(embed_dim * 2, use_bias=use_bias)(inputs)
    if sr_ratio > 1:
        key_value = tf.reshape(key_value, shape=[-1, (height * width) // (sr_ratio ** 2), 2, n_heads, head_dim])
    else:
        key_value = tf.reshape(key_value, shape=[-1, n_patches, 2, n_heads, head_dim])
    key_value = tf.transpose(key_value, perm=[2, 0, 3, 1, 4])
    key, value = key_value[0], key_value[1]

    alpha = tf.matmul(a=query, b=key, transpose_b=True) * scaler
    alpha_prime = tf.nn.softmax(alpha, axis=-1)
    alpha_prime = Dropout(rate=attention_drop_rate)(alpha_prime)

    b = tf.matmul(alpha_prime, value)
    b = tf.transpose(b, perm=[0, 2, 1, 3])
    b = tf.reshape(b, shape=[-1, n_patches, embed_dim])

    x = Dense(embed_dim, use_bias=use_bias)(b)
    x = Dropout(rate=projection_drop_rate)(x)

    return x


def mixed_feedforward_network(inputs, height, width, embed_dim, expansion_rate=4, drop_rate=0., ):
    batches, n_patches, channels = inputs.shape
    assert n_patches == height * width and channels == embed_dim

    x = Dense(int(embed_dim * expansion_rate), use_bias=True)(inputs)
    x = tf.reshape(x, shape=[-1, height, width, int(embed_dim * expansion_rate)])
    x = DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(x)
    x = tf.reshape(x, shape=[-1, n_patches, int(embed_dim * expansion_rate)])
    x = Activation('gelu')(x)
    x = Dense(embed_dim, use_bias=True)(x)
    x = Dropout(rate=drop_rate)(x)

    return x

def seg_former_encoder_block(inputs, height, width, embed_dim, n_heads=8, sr_ratio=1, expansion_rate=4,
                             attention_drop_rate=0., projection_drop_rate=0., drop_rate=0.):
    x = LayerNormalization()(inputs)
    x = efficient_multi_head_attention(x, height, width, embed_dim, n_heads=n_heads, sr_ratio=sr_ratio,
                                       attention_drop_rate=attention_drop_rate,
                                       projection_drop_rate=projection_drop_rate)
    branch1 = Add()([inputs, x])
    x = LayerNormalization()(branch1)
    x = mixed_feedforward_network(x, height, width, embed_dim, expansion_rate, drop_rate)
    x = Add()([branch1, x])

    return x

def SegFormer(n_classes,
              input_height,
              input_width,
              n_blocks=None,
              embed_dims=None,
              decoder_embed_dim=256,
              patch_sizes=None,
              strides=None,
              heads=None,
              reduction_ratios=None,
              expansion_rate=None,
              attention_drop_rate=0.,
              drop_rate=0.,
              name=None,
              ):
    if expansion_rate is None:
        expansion_rate = [8, 8, 4, 4]
    if reduction_ratios is None:
        reduction_ratios = [8, 4, 2, 1]
    if heads is None:
        heads = [1, 2, 4, 8]
    if strides is None:
        strides = [4, 2, 2, 2]
    if patch_sizes is None:
        patch_sizes = [7, 3, 3, 3]
    if embed_dims is None:
        embed_dims = [32, 64, 160, 256]
    if n_blocks is None:
        n_blocks = [2, 2, 2, 2]

    block_range = np.cumsum([0] + n_blocks)
    attention_scheduler = np.linspace(0, attention_drop_rate, num=sum(n_blocks))
    projection_scheduler = np.linspace(0, drop_rate, num=sum(n_blocks))

    shape = (input_height, input_width, 3)
    input_shape = tuple(flatten_tuples(shape))
    inputs = Input(input_shape)

    # encoder
    # stage 1
    x, height1, width1 = overlap_patch_embedding(inputs, embed_dims[0],
                                                 kernel_size=patch_sizes[0], strides=strides[0])

    for index in range(n_blocks[0]):
        attention_range = attention_scheduler[block_range[0]: block_range[1]]
        projection_range = projection_scheduler[block_range[0]: block_range[1]]
        x = seg_former_encoder_block(x, height1, width1, embed_dims[0], heads[0],
                                     sr_ratio=reduction_ratios[0],
                                     expansion_rate=expansion_rate[0],
                                     attention_drop_rate=attention_range[index],
                                     projection_drop_rate=projection_range[index],
                                     drop_rate=drop_rate)
    x = LayerNormalization()(x)
    feature1 = tf.reshape(x, shape=[-1, height1, width1, embed_dims[0]])

    # stage 2
    x, height2, width2 = overlap_patch_embedding(feature1, embed_dims[1],
                                                 kernel_size=patch_sizes[1], strides=strides[1])

    for index in range(n_blocks[1]):
        attention_range = attention_scheduler[block_range[1]: block_range[2]]
        projection_range = projection_scheduler[block_range[1]: block_range[2]]
        x = seg_former_encoder_block(x, height2, width2, embed_dims[1], heads[1],
                                     sr_ratio=reduction_ratios[1],
                                     expansion_rate=expansion_rate[1],
                                     attention_drop_rate=attention_range[index],
                                     projection_drop_rate=projection_range[index],
                                     drop_rate=drop_rate)
    x = LayerNormalization()(x)
    feature2 = tf.reshape(x, shape=[-1, height2, width2, embed_dims[1]])

    # stage 3
    x, height3, width3 = overlap_patch_embedding(feature2, embed_dims[2],
                                                 kernel_size=patch_sizes[2], strides=strides[2])
    for index in range(n_blocks[2]):
        attention_range = attention_scheduler[block_range[2]: block_range[3]]
        projection_range = projection_scheduler[block_range[2]: block_range[3]]
        x = seg_former_encoder_block(x, height3, width3, embed_dims[2], heads[2],
                                     sr_ratio=reduction_ratios[2],
                                     expansion_rate=expansion_rate[2],
                                     attention_drop_rate=attention_range[index],
                                     projection_drop_rate=projection_range[index],
                                     drop_rate=drop_rate)
    x = LayerNormalization()(x)
    feature3 = tf.reshape(x, shape=[-1, height3, width3, embed_dims[2]])

    # stage 4
    x, height4, width4 = overlap_patch_embedding(feature3, embed_dims[3],
                                                 kernel_size=patch_sizes[3], strides=strides[3])
    for index in range(n_blocks[3]):
        attention_range = attention_scheduler[block_range[3]: block_range[4]]
        projection_range = projection_scheduler[block_range[3]: block_range[4]]
        x = seg_former_encoder_block(x, height4, width4, embed_dims[3], heads[3],
                                     sr_ratio=reduction_ratios[3],
                                     expansion_rate=expansion_rate[3],
                                     attention_drop_rate=attention_range[index],
                                     projection_drop_rate=projection_range[index],
                                     drop_rate=drop_rate)
    x = LayerNormalization()(x)
    feature4 = tf.reshape(x, shape=[-1, height4, width4, embed_dims[3]])

    feature1 = seg_former_decoder_block(feature1, decoder_embed_dim, up_size=(4, 4))
    feature2 = seg_former_decoder_block(feature2, decoder_embed_dim, up_size=(8, 8))
    feature3 = seg_former_decoder_block(feature3, decoder_embed_dim, up_size=(16, 16))
    feature4 = seg_former_decoder_block(feature4, decoder_embed_dim, up_size=(32, 32))

    x = seg_former_head([feature1, feature2, feature3, feature4], decoder_embed_dim, n_classes, drop_rate)
    #model = Model(inputs, x, name = "segformer")
    
    model = get_segmentation_model(inputs, x)
    model.model_name = name

    return model


def SegFormerB0(n_classes, input_height, input_width, attention_drop_rate=0., drop_rate=0.):
    return SegFormer(n_classes, input_height, input_width, n_blocks=[2, 2, 2, 2], embed_dims=[32, 64, 120, 256],
                     decoder_embed_dim=256, patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2], heads=[1, 2, 4, 8],
                     reduction_ratios=[8, 4, 2, 1], expansion_rate=[8, 8, 4, 4],
                     attention_drop_rate=attention_drop_rate, drop_rate=drop_rate, name="segformerB0")


def SegFormerB1(n_classes, input_height, input_width, attention_drop_rate=0., drop_rate=0.):
    return SegFormer(n_classes, input_height, input_width, n_blocks=[2, 2, 2, 2], embed_dims=[64, 128, 320, 512],
                     decoder_embed_dim=256, patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2], heads=[1, 2, 4, 8],
                     reduction_ratios=[8, 4, 2, 1], expansion_rate=[8, 8, 4, 4],
                     attention_drop_rate=attention_drop_rate, drop_rate=drop_rate, name="segformerB1")


def SegFormerB2(n_classes, input_height, input_width, attention_drop_rate=0., drop_rate=0.):
    return SegFormer(n_classes, input_height, input_width, n_blocks=[3, 3, 6, 3], embed_dims=[64, 128, 320, 512],
                     decoder_embed_dim=768, patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2], heads=[1, 2, 4, 8],
                     reduction_ratios=[8, 4, 2, 1], expansion_rate=[8, 8, 4, 4],
                     attention_drop_rate=attention_drop_rate, drop_rate=drop_rate, name="segformerB2")


def SegFormerB3(n_classes, input_height, input_width, attention_drop_rate=0., drop_rate=0.):
    return SegFormer(n_classes, input_height, input_width, n_blocks=[3, 3, 18, 3], embed_dims=[64, 128, 320, 512],
                     decoder_embed_dim=768, patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2], heads=[1, 2, 4, 8],
                     reduction_ratios=[8, 4, 2, 1], expansion_rate=[8, 8, 4, 4],
                     attention_drop_rate=attention_drop_rate, drop_rate=drop_rate, name="segformerB3")


def SegFormerB4(n_classes, input_height, input_width, attention_drop_rate=0., drop_rate=0.):
    return SegFormer(n_classes, input_height, input_width, n_blocks=[3, 8, 27, 3], embed_dims=[64, 128, 320, 512],
                     decoder_embed_dim=768, patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2], heads=[1, 2, 4, 8],
                     reduction_ratios=[8, 4, 2, 1], expansion_rate=[8, 8, 4, 4],
                     attention_drop_rate=attention_drop_rate, drop_rate=drop_rate, name="segformerB4")


def SegFormerB5(n_classes, input_height, input_width, attention_drop_rate=0., drop_rate=0.):
    return SegFormer(n_classes, input_height, input_width, n_blocks=[3, 6, 40, 3], embed_dims=[64, 128, 320, 512],
                     decoder_embed_dim=768, patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2], heads=[1, 2, 4, 8],
                     reduction_ratios=[8, 4, 2, 1], expansion_rate=[4, 4, 4, 4],
                     attention_drop_rate=attention_drop_rate, drop_rate=drop_rate, name="segformerB5")