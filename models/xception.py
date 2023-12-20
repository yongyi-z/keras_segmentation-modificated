import keras
import tensorflow
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import layers
from tensorflow.keras.utils import get_source_inputs, get_file
from tensorflow.keras import backend as K
from tensorflow.keras.applications.xception import Xception
#from keras.applications.imagenet_utils import decode_predictions
#from keras.applications.imagenet_utils import _obtain_input_shape
from .config import IMAGE_ORDERING

TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5'

TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'

def get_Xception_encoder(input_height=299,  input_width=299, include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000):

    #assert input_height % 32 == 0

    #assert input_width % 32 == 0

    if IMAGE_ORDERING == 'channels_first':
        img_input = Input(shape=(3, input_height, input_width))

    elif IMAGE_ORDERING == 'channels_last':
        img_input = Input(shape=(input_height, input_width, 3))

    if IMAGE_ORDERING == 'channels_last':
        bn_axis = 3

    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3), data_format=IMAGE_ORDERING)(img_input)
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(x)
    #f1 = x

    x = BatchNormalization(axis=bn_axis, name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = BatchNormalization(axis=bn_axis, name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)

    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization(axis=bn_axis)(residual)

    f1 = x

    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
    x = BatchNormalization(axis=bn_axis, name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
    x = BatchNormalization(axis=bn_axis, name='block2_sepconv2_bn')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)

    x = layers.add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization(axis=bn_axis)(residual)

    f2 = x

    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
    x = BatchNormalization(axis=bn_axis, name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
    x = BatchNormalization(axis=bn_axis, name='block3_sepconv2_bn')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
    x = layers.add([x, residual])

    residual = Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization(axis=bn_axis)(residual)

    f3 = x

    x = Activation('relu', name='block4_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
    x = BatchNormalization(axis=bn_axis, name='block4_sepconv1_bn')(x)
    x = Activation('relu', name='block4_sepconv2_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
    x = BatchNormalization(axis=bn_axis, name='block4_sepconv2_bn')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
    x = layers.add([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
        x = BatchNormalization(axis=bn_axis, name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
        x = BatchNormalization(axis=bn_axis, name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
        x = BatchNormalization(axis=bn_axis, name=prefix + '_sepconv3_bn')(x)
        x = layers.add([x, residual])

    residual = Conv2D(1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization(axis=bn_axis)(residual)

    f4 = x

    x = Activation('relu', name='block13_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
    x = BatchNormalization(axis=bn_axis, name='block13_sepconv1_bn')(x)
    x = Activation('relu', name='block13_sepconv2_act')(x)
    x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
    x = BatchNormalization(axis=bn_axis, name='block13_sepconv2_bn')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)

    x = layers.add([x, residual])

    x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
    x = BatchNormalization(axis=bn_axis, name='block14_sepconv1_bn')(x)
    x = Activation('relu', name='block14_sepconv1_act')(x)
    x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
    x = BatchNormalization(axis=bn_axis, name='block14_sepconv2_bn')(x)
    x = Activation('relu', name='block14_sepconv2_act')(x)

    f5 = x

    x = AveragePooling2D((7, 7), data_format=IMAGE_ORDERING, name='avg_pool')(x)

    # load weights

    if weights == 'imagenet':

        if include_top:

            weights_path = get_file('xception_weights_tf_dim_ordering_tf_kernels.h5', TF_WEIGHTS_PATH, cache_subdir='models')
        else:
            weights_path = get_file('xception_weights_tf_dim_ordering_tf_kernels_notop.h5', TF_WEIGHTS_PATH_NO_TOP, cache_subdir='models')

        Model(img_input, x).load_weights(weights_path, by_name = True, skip_mismatch = True)
    return img_input, [f1, f2, f3, f4, f5]


IMAGE_ORDERING = 'channels_last'
if IMAGE_ORDERING == 'channels_last':
    bn_axis = 3
else:
    bn_axis = 1

if IMAGE_ORDERING == 'channels_first':
    MERGE_AXIS = 1
elif IMAGE_ORDERING == 'channels_last':
    MERGE_AXIS = -1

def densep(input, input_filter, dilate=1):
    pw1 = Conv2D(filters = input_filter/2, kernel_size= 1, strides= 1, dilation_rate = dilate)(input)
    pw1 = BatchNormalization(axis=bn_axis)(pw1)
    #print(pw1.shape)
    dw1 = DepthwiseConv2D(kernel_size= 3, strides= 1, padding= 'same', dilation_rate= dilate, depthwise_regularizer=None)(pw1)
    dw1 = BatchNormalization(axis=bn_axis)(dw1)
    #print(dw1.shape)
    concat1 = concatenate([dw1, input], axis=MERGE_AXIS)
    #print(concat1.shape)

    pw2 = Conv2D(filters = input_filter/2, kernel_size= 1, strides= 1, dilation_rate = dilate)(concat1)
    pw2 = BatchNormalization(axis=bn_axis)(pw1)
    #print(pw2.shape)
    dw2 = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', dilation_rate=dilate, depthwise_regularizer=None)(pw2)
    dw2 = BatchNormalization(axis=bn_axis)(dw1)
    #print(dw2.shape)
    concat2 = concatenate([dw2,dw1, input], axis=MERGE_AXIS)
    #print(concat2.shape)

    pw3 = Conv2D(filters=input_filter / 2, kernel_size=1, strides=1, dilation_rate=dilate)(concat2)
    pw3 = BatchNormalization(axis=bn_axis)(pw3)
    #print(pw3.shape)
    dw3 = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', dilation_rate=dilate, depthwise_regularizer=None)(pw3)
    dw3 = BatchNormalization(axis=bn_axis)(dw3)
    #print(dw3.shape)
    concat3 = concatenate([dw3, dw2, dw1, input], axis=MERGE_AXIS)
    #print(concat3.shape)

    pw4 = Conv2D(filters=input_filter, kernel_size=1, strides=1, dilation_rate=dilate)(concat3)
    pw4 = BatchNormalization(axis=bn_axis)(pw4)
    #print(pw4.shape)
    dw4 = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', dilation_rate=dilate, depthwise_regularizer=None)(pw4)
    dw4 = BatchNormalization(axis=bn_axis)(dw4)
    #print(dw4.shape)
    return dw4

def spatial_pyramid(input, output_filter):
    input_filter = output_filter/4
    d1 = densep(input,input_filter,1)
    d2 = densep(input,input_filter,2)
    d4 = densep(input,input_filter,4)
    d8 = densep(input,input_filter,8)

    add1 = d2
    add2 = add1 + d4
    add3 = add2 + d8

    concat = concatenate([d1, add1, add2, add3],MERGE_AXIS)
    combine = input + concat
    output = BatchNormalization(axis=bn_axis)(combine)
    return output



def get_Xception_sp_encoder(input_height=512,  input_width=512):
    if IMAGE_ORDERING == 'channels_first':
        img_input = Input(shape=(3, input_height, input_width))
    elif IMAGE_ORDERING == 'channels_last':
        img_input = Input(shape=(input_height, input_width, 3))

    model = Xception(weights="imagenet")
    #model.summary()
    arr = []
    #counter = 0
    for i in model.layers:
        if i.name.find('add') == -1 and i.name.find('input') == -1:
            arr.append(i)
            #print(counter, i.name)
            #counter+=1
        if i.name == 'add_2':
            break

    x = ZeroPadding2D((3, 3), data_format=IMAGE_ORDERING)(img_input)
    # block 1
    x = arr[0](x)
    x = arr[1](x)
    x = arr[2](x)
    x = arr[3](x)
    x = arr[4](x)
    x = arr[5](x)
    #print(arr[5].name)

    # # side pyramid
    sp = arr[11](x)
    sp = arr[13](sp)
    sp = spatial_pyramid(sp, sp.shape[3])
    #print(sp.shape)
    f1 = sp

    # block 2
    x = arr[6](x)
    x = arr[7](x)
    x = arr[8](x)
    x = arr[9](x)
    x = arr[10](x)
    x = arr[12](x)
    x = x + sp
    #print(x.shape)

    # side pyramid
    sp = arr[20](x)
    sp = arr[22](sp)
    sp = spatial_pyramid(sp, sp.shape[3])
    #print(sp.shape)
    f2 = sp

    # block 3
    x = arr[14](x)
    x = arr[15](x)
    x = arr[16](x)
    x = arr[17](x)
    x = arr[18](x)
    x = arr[19](x)
    x = arr[21](x)
    x = x + sp
    #print(x.shape)

    # side pyramid
    sp = arr[29](x)
    sp = arr[31](sp)
    sp = spatial_pyramid(sp, sp.shape[3])
    #print(sp.shape)

    # block 4
    x = arr[23](x)
    x = arr[24](x)
    x = arr[25](x)
    x = arr[26](x)
    x = arr[27](x)
    x = arr[28](x)
    x = arr[30](x)
    x = x + sp
    #print(x.shape)
    #print(sp.shape)
    return img_input, [f1, f2, x]
