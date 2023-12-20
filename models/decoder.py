from tensorflow.keras.models import *
from tensorflow.keras.layers import *

from .config import IMAGE_ORDERING
from .model_utils import get_segmentation_model
from .xception import *

if IMAGE_ORDERING == 'channels_last':
    bn_axis = 3
else:
    bn_axis = 1

if IMAGE_ORDERING == 'channels_first':
    MERGE_AXIS = 1
elif IMAGE_ORDERING == 'channels_last':
    MERGE_AXIS = -1
    
def decoder(n_classes, encoder, input_height=512,  input_width=512):
    img_input, levels = encoder(input_height=input_height, input_width=input_width)
    [f1, f2, x] = levels
    #o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(x)
    o = Conv2DTranspose(filters=256, kernel_size=2, strides=2, data_format=IMAGE_ORDERING)(x)
    o = BatchNormalization(axis=bn_axis)(o)
    pw = Conv2D(filters=256, kernel_size=1, strides=1)(o)
    pw = BatchNormalization(axis=bn_axis)(pw)
    pw = concatenate([pw, f2], axis=MERGE_AXIS)
    dw = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', depthwise_regularizer=None)(pw)
    dw = BatchNormalization(axis=bn_axis)(dw)

    o = Conv2DTranspose(filters=128, kernel_size=2, strides=2, data_format=IMAGE_ORDERING)(dw)
    o = BatchNormalization(axis=bn_axis)(o)
    pw = Conv2D(filters=128, kernel_size=1, strides=1)(o)
    pw = BatchNormalization(axis=bn_axis)(pw)
    pw = concatenate([pw,f1], axis=MERGE_AXIS)
    dw = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', depthwise_regularizer=None)(pw)
    dw = BatchNormalization(axis=bn_axis)(dw)

    o = Conv2DTranspose(filters=64, kernel_size=2, strides=2, data_format=IMAGE_ORDERING)(dw)
    o = BatchNormalization(axis=bn_axis)(o)
    pw = Conv2D(filters=64, kernel_size=1, strides=1)(o)
    pw = BatchNormalization(axis=bn_axis)(pw)
    dw = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', depthwise_regularizer=None)(pw)
    dw = BatchNormalization(axis=bn_axis)(dw)

    o = Conv2DTranspose(filters=n_classes, kernel_size=2, strides=2, data_format=IMAGE_ORDERING)(dw)
    o = BatchNormalization(axis=bn_axis)(o)
    pw = Conv2D(filters=n_classes, kernel_size=1, strides=1)(o)
    pw = BatchNormalization(axis=bn_axis)(pw)
    dw = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', depthwise_regularizer=None)(pw)
    dw = BatchNormalization(axis=bn_axis)(dw)

    return img_input, dw

def xception_sp_pw_dw(n_classes, input_height, input_width):
    img_input, o = decoder(n_classes, get_Xception_sp_encoder, input_height=input_height, input_width=input_width)
    model = get_segmentation_model(img_input, o)
    model.model_name = "xception_sp_pw_dw"
    return model
    
    
