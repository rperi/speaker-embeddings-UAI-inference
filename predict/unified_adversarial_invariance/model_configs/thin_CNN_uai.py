import keras
from keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from model_configs.core import ModelConfigBase
from utils.training_utils \
    import disentanglement_loss

from keras.layers import BatchNormalization
from keras.layers import UpSampling2D
from keras.layers import Concatenate
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.regularizers import l2

weight_decay = 1e-4
bn_axis = 3
drp_out = 0.2

# Based on the architecture in https://arxiv.org/pdf/1803.05307.pdf
# (DEEP CNN BASED FEATURE EXTRACTOR FOR TEXT-PROMPTED SPEAKERRECOGNITION)


class ModelConfig(ModelConfigBase):

    def __init__(self):

        super(ModelConfig, self).__init__()
        self.nclasses = 7323
        self.nz = 2
        self.x_shape = (80, 400, 1)  # log-mel FB energies of 4 sec segment
        self.embedding_dim_1 = 128  # Change this later
        self.embedding_dim_2 = 128  # Change this later
        self.nz = 'tanh'
        self.predictor_loss = 'categorical_crossentropy'
        self.decoder_loss = 'mean_squared_error'
        self.disentangler_loss = disentanglement_loss

    def encoder(self, name='encoder'):

        x_in = Input(self.x_shape, name='encoder_input')

        #-------------------
        # Thin CNN endoder block
        #-------------------

        x = Conv2D(filters=128, kernel_size=(7, 7),
               trainable=True,
               kernel_regularizer=l2(weight_decay),
               padding='same'
               )(x_in)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), (2, 2))(x)

        x = conv_block(x, [128, 192], [(1, 1), (5, 5)])
        x = MaxPooling2D((2, 2), (2, 2))(x)

        x = conv_block(x, [192, 256], [(1, 1), (5, 5)])
        x = MaxPooling2D((2, 2), (2, 2))(x)

        x = conv_block(x, [256, 128], [(1, 1), (3, 3)])

        x = conv_block(x, [128, 128], [(1, 1), (3, 3)])

        x = MaxPooling2D((2, 2), (2, 5))(x)

        #x = Flatten()(x)
        x = GlobalAveragePooling2D()(x)

        #-------------------------
        # Fully Connected Block 1
        #-------------------------

        bottleneck_dim = 256
        h = Dense(bottleneck_dim, name='bottleneck', activation='relu', kernel_initializer='orthogonal',
                   use_bias=True, trainable=True, kernel_regularizer=keras.regularizers.l2(weight_decay),
                   bias_regularizer=keras.regularizers.l2(weight_decay))(x)

        # -------------------------
        # Embedding Block 1
        # -------------------------

        e1 = Dense(self.embedding_dim_1, name='embedding_1', activation=self.nz, kernel_initializer='orthogonal',
                   use_bias=True, trainable=True, kernel_regularizer=keras.regularizers.l2(weight_decay),
                           bias_regularizer=keras.regularizers.l2(weight_decay))(h)
        e2 = Dense(self.embedding_dim_2, name='embedding_2', activation=self.nz, kernel_initializer='orthogonal',
                   use_bias=True, trainable=True, kernel_regularizer=keras.regularizers.l2(weight_decay),
                           bias_regularizer=keras.regularizers.l2(weight_decay))(h)

        m = Model(inputs=[x_in], outputs=[e1, e2], name=name)
        m.summary()
        return m

    def noisy_transformer(self, params=[0.5], name='noisy_transformer'):
        dropout_rate = params[0]
        return Dropout(dropout_rate)

    def predictor(self, name='predictor'):
        e1 = Input((self.embedding_dim_1,), name='predictor_input')
        h = BatchNormalization(name='predictor_bn1')(e1)

        h = Dense(256, name='predictor_fc2')(h)
        h = BatchNormalization(name='predictor_bn2')(h)
        h = Activation('relu', name='predictor_relu2')(h)

        h = Dense(512, name='predictor_fc3')(h)
        h = BatchNormalization(name='predictor_bn3')(h)
        h = Activation('relu', name='predictor_relu3')(h)

        y = Dense(self.nclasses, activation='softmax', name='predictor_output')(h)

        return Model(e1, y, name=name)

    def decoder(self, name='decoder'):

        x_height, x_width, x_channels = self.x_shape
        x_height1 = int(x_height / 16)
        x_width1 = int(x_width / 16)

        e1 = Input((self.embedding_dim_1,))
        e2 = Input((self.embedding_dim_2,))
        e = Concatenate()([e1, e2])

        h = Dense(int(x_height1 * x_width1 * 128), name='shape_match')(e)
        h = BatchNormalization(name='decoder_bn1')(h)
        h = Activation('relu', name='decoder_relu0')(h)
        h = Reshape((x_height1, x_width1, 128))(h)

        h = UpSampling2D(size=(2, 2))(h)
        h = conv_block(h, [128, 128], [(3, 3), (1, 1)])

        h = conv_block(h, [128, 256], [(3, 3), (1, 1)])

        h = UpSampling2D(size=(2, 2))(h)
        h = conv_block(h, [256, 192], [(5, 5), (1, 1)])

        h = UpSampling2D(size=(2, 2))(h)
        h = conv_block(h, [192, 128], [(5, 5), (1, 1)])

        h = UpSampling2D(size=(2, 2))(h)
        x = Conv2D(filters=128, kernel_size=(7, 7),
                   trainable=True,
                   kernel_regularizer=l2(weight_decay),
                   padding='same'
                   )(h)
        x = Activation('relu')(x)

        x = Conv2D(x_channels, (1, 1),
                    kernel_initializer='orthogonal',
                    use_bias=False, trainable=True,
                    kernel_regularizer=l2(weight_decay),
                    padding='same',
                    name='decoder_conv3')(x)

        x = Activation('linear')(x)   # RP: have to change this. Can't use sigmoid if input is log mel. Probably linear activation
        x_out = Reshape(self.x_shape, name='decoder_output')(x)

        m = Model(inputs=[e1, e2], outputs=[x_out], name=name)
        m.summary()
        return m

    def disentangler(self, input_dim=None, output_dim=None, name='disentangler'):
        if input_dim is None:
            input_dim = self.embedding_dim_2
        if output_dim is None:
            output_dim = self.embedding_dim_1

        ei = Input((input_dim,), name='disentangler_input')
        ej = Dense(
            output_dim, activation=self.nz,
            name='disentangler_output'
        )(ei)

        return Model(ei, ej, name=name)


def conv_block(x_in, filters, kernels):
    x1 = Conv2D(filters=filters[0], kernel_size=kernels[0],
                trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same'
                )(x_in)
    x1 = BatchNormalization(axis=bn_axis, trainable=True)(x1)
    x1 = Dropout(drp_out)(x1)
    x1 = Activation('relu')(x1)

    x1 = Conv2D(filters=filters[1], kernel_size=kernels[1],
                trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same'
                )(x1)
    x1 = BatchNormalization(axis=bn_axis, trainable=True)(x1)
    x1 = Dropout(drp_out)(x1)
    x1 = Activation('relu')(x1)

    return x1