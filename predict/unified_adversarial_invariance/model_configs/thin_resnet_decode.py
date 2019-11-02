from __future__ import print_function
from __future__ import absolute_import

from keras import layers
from keras.regularizers import l2
from keras.layers import Activation, Conv1D, Conv2D, Input, Lambda
from keras.layers import BatchNormalization, Flatten, Dense, Reshape
from keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import UpSampling2D

weight_decay = 1e-4


def identity_block_2D(input_tensor, kernel_size, filters, stage, block, trainable=True):
    """The identity block is the block that has no deconv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle deconv layer at main path
        filters: list of integers, the filterss of 3 deconv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3

    deconv_name_1 = 'deconv' + str(stage) + '_' + str(block) + '_1x1_reduce'
    bn_name_1 = 'deconv' + str(stage) + '_' + str(block) + '_1x1_reduce/bn'
    x = Conv2D(filters1, (1, 1),
               kernel_initializer='orthogonal',
               use_bias=False,
               trainable=trainable,
               kernel_regularizer=l2(weight_decay),
               name=deconv_name_1)(input_tensor)
    x = BatchNormalization(axis=bn_axis, trainable=trainable, name=bn_name_1)(x)
    x = Activation('relu')(x)

    deconv_name_2 = 'deconv' + str(stage) + '_' + str(block) + '_3x3'
    bn_name_2 = 'deconv' + str(stage) + '_' + str(block) + '_3x3/bn'
    x = Conv2D(filters2, kernel_size,
               padding='same',
               kernel_initializer='orthogonal',
               use_bias=False,
               trainable=trainable,
               kernel_regularizer=l2(weight_decay),
               name=deconv_name_2)(x)
    x = BatchNormalization(axis=bn_axis, trainable=trainable, name=bn_name_2)(x)
    x = Activation('relu')(x)

    deconv_name_3 = 'deconv' + str(stage) + '_' + str(block) + '_1x1_increase'
    bn_name_3 = 'deconv' + str(stage) + '_' + str(block) + '_1x1_increase/bn'
    x = Conv2D(filters3, (1, 1),
               kernel_initializer='orthogonal',
               use_bias=False,
               trainable=trainable,
               kernel_regularizer=l2(weight_decay),
               name=deconv_name_3)(x)
    x = BatchNormalization(axis=bn_axis, trainable=trainable, name=bn_name_3)(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def deconv_block_2D(input_tensor, kernel_size, filters, stage, block, strides=(1, 1), trainable=True):
    """A block that has a deconv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle deconv layer at main path
        filters: list of integers, the filterss of 3 deconv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first deconv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3

    deconv_name_1 = 'deconv' + str(stage) + '_' + str(block) + '_1x1_reduce'
    bn_name_1 = 'deconv' + str(stage) + '_' + str(block) + '_1x1_reduce/bn'
    x = Conv2D(filters1, (1, 1),
               strides=strides,
               kernel_initializer='orthogonal',
               use_bias=False,
               trainable=trainable,
               kernel_regularizer=l2(weight_decay),
               name=deconv_name_1)(input_tensor)
    x = BatchNormalization(axis=bn_axis, trainable=trainable, name=bn_name_1)(x)
    x = Activation('relu')(x)

    deconv_name_2 = 'deconv' + str(stage) + '_' + str(block) + '_3x3'
    bn_name_2 = 'deconv' + str(stage) + '_' + str(block) + '_3x3/bn'
    x = Conv2D(filters2, kernel_size, padding='same',
               kernel_initializer='orthogonal',
               use_bias=False,
               trainable=trainable,
               kernel_regularizer=l2(weight_decay),
               name=deconv_name_2)(x)
    x = BatchNormalization(axis=bn_axis, trainable=trainable, name=bn_name_2)(x)
    x = Activation('relu')(x)

    deconv_name_3 = 'deconv' + str(stage) + '_' + str(block) + '_1x1_increase'
    bn_name_3 = 'deconv' + str(stage) + '_' + str(block) + '_1x1_increase/bn'
    x = Conv2D(filters3, (1, 1),
               kernel_initializer='orthogonal',
               use_bias=False,
               trainable=trainable,
               kernel_regularizer=l2(weight_decay),
               name=deconv_name_3)(x)
    x = BatchNormalization(axis=bn_axis, trainable=trainable, name=bn_name_3)(x)

    deconv_name_4 = 'deconv' + str(stage) + '_' + str(block) + '_1x1_proj'
    bn_name_4 = 'deconv' + str(stage) + '_' + str(block) + '_1x1_proj/bn'
    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      kernel_initializer='orthogonal',
                      use_bias=False,
                      trainable=trainable,
                      kernel_regularizer=l2(weight_decay),
                      name=deconv_name_4)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, trainable=trainable, name=bn_name_4)(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

# class Unpooling2D(Layer):
#     def __init__(self, poolsize=(2, 2), ignore_border=True):
#         super(Unpooling2D,self).__init__()
#         self.input = T.tensor4()
#         self.poolsize = poolsize
#         self.ignore_border = ignore_border
#
#     def get_output(self, train):
#         X = self.get_input(train)
#         s1 = self.poolsize[0]
#         s2 = self.poolsize[1]
#         output = X.repeat(s1, axis=2).repeat(s2, axis=3)
#         return output
#
#     def get_config(self):
#         return {"name":self.__class__.__name__,
#             "poolsize":self.poolsize,
#             "ignore_border":self.ignore_border}

def resnet_2D_v1(inputs, mode='train'):

    bn_axis = 3
    #if mode == 'train':
      #  inputs = Input(shape=input_dim, name='input')
    #else:
        #inputs = Input(shape=(input_dim[0], None, input_dim[-1]), name='input')

    x1 = UpSampling2D(size=(2, 2))(inputs)

    x1 = deconv_block_2D(x1, 3, [96, 48, 48], stage=1, block='c', trainable=True)
    x1 = identity_block_2D(x1, 3, [96, 48, 48], stage=1, block='a', trainable=True)
    x1 = identity_block_2D(x1, 3, [96, 48, 48], stage=1, block='b', trainable=True)

    x2 = deconv_block_2D(x1, 3, [48, 28, 28], stage=2, block='c', trainable=True)
    x2 = identity_block_2D(x2, 3, [48, 28, 28], stage=2, block='a', trainable=True)
    x2 = identity_block_2D(x2, 3, [48, 28, 28], stage=2, block='b', trainable=True)

    x3 = deconv_block_2D(x2, 3, [28, 16, 16], stage=3, block='c', trainable=True)
    x3 = identity_block_2D(x3, 3, [28, 16, 16], stage=3, block='a', trainable=True)
    x3 = identity_block_2D(x3, 3, [28, 16, 16], stage=3, block='b', trainable=True)

    x4 = deconv_block_2D(x3, 3, [16, 8, 8], stage=4, block='b', strides=(1, 1), trainable=True)
    x4 = identity_block_2D(x4, 3, [16, 8, 8], stage=4, block='a', trainable=True)

    x5 = UpSampling2D(size=(2, 2))(x4)

    x5 = Conv2D(8, (7, 7),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='deconv1_1/3x3_s1')(x5)

    x5 = BatchNormalization(axis=bn_axis, name='deconv1_1/3x3_s1/bn', trainable=True)(x5)
    x5 = Activation('relu')(x5)

    return inputs, x5

    #
    # # ===============================================
    # #            deconvolution Block 1
    # # ===============================================
    # x1 = Conv2D(64, (7, 7),
    #             kernel_initializer='orthogonal',
    #             use_bias=False, trainable=True,
    #             kernel_regularizer=l2(weight_decay),
    #             padding='same',
    #             name='deconv1_1/3x3_s1')(inputs)
    #
    # x1 = BatchNormalization(axis=bn_axis, name='deconv1_1/3x3_s1/bn', trainable=True)(x1)
    # x1 = Activation('relu')(x1)
    # x1 = MaxPooling2D((2, 2), strides=(2, 2))(x1)
    #
    # # ===============================================
    # #            deconvolution Section 2
    # # ===============================================
    # x2 = deconv_block_2D(x1, 3, [96, 48, 48], stage=2, block='a', strides=(1, 1), trainable=True)
    # x2 = identity_block_2D(x2, 3, [96, 48, 48], stage=2, block='b', trainable=True)
    #
    # # ===============================================
    # #            deconvolution Section 3
    # # ===============================================
    # x3 = deconv_block_2D(x2, 3, [128, 96, 96], stage=3, block='a', trainable=True)
    # x3 = identity_block_2D(x3, 3, [128, 96, 96], stage=3, block='b', trainable=True)
    # x3 = identity_block_2D(x3, 3, [128, 96, 96], stage=3, block='c', trainable=True)
    # # ===============================================
    # #            deconvolution Section 4
    # # ===============================================
    # x4 = deconv_block_2D(x3, 3, [256, 128, 128], stage=4, block='a', trainable=True)
    # x4 = identity_block_2D(x4, 3, [256, 128, 128], stage=4, block='b', trainable=True)
    # x4 = identity_block_2D(x4, 3, [256, 128, 128], stage=4, block='c', trainable=True)
    # # ===============================================
    # #            deconvolution Section 5
    # # ===============================================
    # x5 = deconv_block_2D(x4, 3, [512, 256, 256], stage=5, block='a', trainable=True)
    # x5 = identity_block_2D(x5, 3, [512, 256, 256], stage=5, block='b', trainable=True)
    # x5 = identity_block_2D(x5, 3, [512, 256, 256], stage=5, block='c', trainable=True)
    # y = MaxPooling2D((3, 1), strides=(2, 1), name='mpool2')(x5)

    # return inputs, y

