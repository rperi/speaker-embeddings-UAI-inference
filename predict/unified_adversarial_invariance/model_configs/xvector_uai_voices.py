from model_configs.core import ModelConfigBase
from utils.training_utils \
    import disentanglement_loss
from model_configs import thin_resnet_encode
from model_configs import thin_resnet_decode

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
import keras

weight_decay = 1e-4


class ModelConfig(ModelConfigBase):

    def __init__(self):

        super(ModelConfig, self).__init__()
        self.nclasses = 200
        self.x_shape = (512, )  # log-mel FB energies of 4 sec segment
        self.embedding_dim_1 = 128  # Change this later
        self.embedding_dim_2 = 128  # Change this later
        self.nz = 'tanh'
        self.predictor_loss = 'sparse_categorical_crossentropy'
        self.decoder_loss = 'mean_squared_error'
        self.disentangler_loss = disentanglement_loss

    def encoder(self, name='encoder'):

        x_in = Input(self.x_shape, name='encoder_input')
        
        x = x_in
        x = Dense(256, name='enc_fc1', activation='relu',
                        use_bias=True, trainable=True, kernel_regularizer=keras.regularizers.l2(weight_decay),
                        bias_regularizer=keras.regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)

        #x = Dense(256, name='enc_fc2', activation='relu',
        #        use_bias=True, trainable=True, kernel_regularizer=keras.regularizers.l2(weight_decay),
        #        bias_regularizer=keras.regularizers.l2(weight_decay))(x)
        #x = BatchNormalization()(x)
        #x = Dropout(0.2)(x)

        e1 = Dense(self.embedding_dim_1, name='embedding_1', activation=self.nz, kernel_initializer='orthogonal',
                   use_bias=True, trainable=True, kernel_regularizer=keras.regularizers.l2(weight_decay),
                   bias_regularizer=keras.regularizers.l2(weight_decay))(x)
        e2 = Dense(self.embedding_dim_2, name='embedding_2', activation=self.nz, kernel_initializer='orthogonal',
                   use_bias=True, trainable=True, kernel_regularizer=keras.regularizers.l2(weight_decay),
                   bias_regularizer=keras.regularizers.l2(weight_decay))(x)

        m = Model(inputs=[x_in], outputs=[e1, e2], name=name)
        m.summary()
        return m

    def noisy_transformer(self, params=[0.5], name='noisy_transformer'):
        dropout_rate = params[0]
        return Dropout(dropout_rate)

    def predictor(self, name='predictor'):
        e1 = Input((self.embedding_dim_1,), name='predictor_input')
        h = e1 #BatchNormalization(name='predictor_bn1')(e1)

        h = Dense(128, name='predictor_fc1')(h)
        h = BatchNormalization(name='predictor_bn1')(h)
        h = Activation('relu', name='predictor_relu1')(h)
        h = Dropout(0.4)(h)

        #h = Dense(512, name='predictor_fc3')(h)
        #h = BatchNormalization(name='predictor_bn3')(h)
        #h = Activation('relu', name='predictor_relu3')(h)
        #h = Dropout(0.2)(h)

        y = Dense(self.nclasses, activation='softmax', name='predictor_output')(h)

        return Model(e1, y, name=name)

    def decoder(self, name='decoder'):

        e1 = Input((self.embedding_dim_1,))
        e2 = Input((self.embedding_dim_2,))
        e = Concatenate()([e1, e2])
        
        h = e
        #h = Dense(512, name='dec_fc1',
        #        use_bias=True, trainable=True, kernel_regularizer=keras.regularizers.l2(weight_decay),
        #        bias_regularizer=keras.regularizers.l2(weight_decay))(h)
        #h = BatchNormalization(name='decoder_bn1')(h)
        #h = Activation('relu', name='decoder_relu1')(h)

        h = Dense(256, name='dec_fc2',
                  use_bias=True, trainable=True, kernel_regularizer=keras.regularizers.l2(weight_decay),
                  bias_regularizer=keras.regularizers.l2(weight_decay))(h)
        h = BatchNormalization(name='decoder_bn2')(h)
        h = Activation('relu', name='decoder_relu2')(h)

        x = Dense(self.x_shape[0], name='dec_fc3',
                  use_bias=True, trainable=True, kernel_regularizer=keras.regularizers.l2(weight_decay),
                  bias_regularizer=keras.regularizers.l2(weight_decay))(h)
        x = Activation('linear')(x)
        x_out = Reshape(self.x_shape,name='decoder_output')(x)

        m = Model(inputs=[e1, e2], outputs=[x_out], name=name)
        m.summary()
        return m

    def disentangler(self, input_dim=None, output_dim=None, name='disentangler'):
        if input_dim is None:
            input_dim = self.embedding_dim_2
        if output_dim is None:
            output_dim = self.embedding_dim_1

        ei = Input((input_dim,), name='disentangler_input')
        
        e = ei
        #e = Dense(self.embedding_dim_1, 
        #          activation='relu', 
        #          name='dis_hidden1')(ei)
        #e = Dropout(0.2)(e)

        #e = Dense(self.embedding_dim_1, 
        #          activation='relu', 
        #          name='dis_hidden2')(e)
        #e = Dropout(0.2)(e)
        
        ej = Dense(
            output_dim, activation=self.nz,
            name='disentangler_output'
        )(e)

        return Model(ei, ej, name=name)
