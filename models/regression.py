#project
from utils import is_model_saved, load_model
# 3p
import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Activation
from keras.layers import Conv2D, UpSampling2D, Reshape, Activation, Input, BatchNormalization, Dense, Concatenate, Add, Flatten, Permute, RepeatVector
from keras.layers import Activation, InputLayer
from keras.callbacks import TensorBoard
from keras import optimizers
from keras import losses


class Regression:

  def __init__(self, load, optimizer='adam', loss=losses.mean_squared_error):
    self.load_saved = load
    # We used the recommended parameters
    self.optimizer = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    self.loss = loss
    self.callbacks = []
    self.name = "regression"
    self.input_type = "reg"

    # Define model
    input_image = Input(shape=(256, 256, 1))

    # Low-Level Features network

    ## Layer 1
    x = Conv2D(64, (3,3), strides=(2,2), activation='relu', padding='same', name='conv_1')(input_image)
    x = BatchNormalization()(x)
    ## Layer 2
    x = Conv2D(128, (3,3), strides=(1,1), activation='relu', padding='same', name='conv_2')(x)
    x = BatchNormalization()(x)
    ## Layer 3
    x = Conv2D(128, (3,3), strides=(2,2), activation='relu', padding='same', name='conv_3')(x)
    x = BatchNormalization()(x)
    ## Layer 4
    x = Conv2D(256, (3,3), strides=(1,1), activation='relu', padding='same', name='conv_4')(x)
    x = BatchNormalization()(x)
    ## Layer 5
    x = Conv2D(256, (3,3), strides=(2,2), activation='relu', padding='same', name='conv_5')(x)
    x = BatchNormalization()(x)
    ## Layer 6
    x = Conv2D(512, (3,3), strides=(1,1), activation='relu', padding='same', name='conv_6')(x)
    x = BatchNormalization()(x)

    # Global Features network

    ## Layer 7
    y = Conv2D(512, (3,3), strides=(2,2), activation='relu', padding='same', name='conv_7')(x)
    y = BatchNormalization()(y)
    ## Layer 8
    y = Conv2D(512, (3,3), strides=(1,1), activation='relu', padding='same', name='conv_8')(y)
    y = BatchNormalization()(y)
    ## Layer 9
    y = Conv2D(512, (3,3), strides=(2,2), activation='relu', padding='same', name='conv_9')(y)
    y = BatchNormalization()(y)
    ## Layer 10
    y = Conv2D(512, (3,3), strides=(1,1), activation='relu', padding='same', name='conv_10')(y)
    y = BatchNormalization()(y)
    ## Flatenning
    y = Flatten()(y)
    ## Layer 11
    y = Dense(1024, activation='relu', name='fc_1')(y)
    y = BatchNormalization()(y)
    ## Layer 12
    y = Dense(512, activation='relu', name='fc_2')(y)
    y = BatchNormalization()(y)
    ## Layer 13
    y = Dense(256, activation='relu', name='fc_3')(y)
    y = BatchNormalization()(y)

    # Mid-Level features network

    ## Layer 14
    z = Conv2D(512, (3,3), strides=(1,1), activation='relu', padding='same', name='conv_11')(x)
    z = BatchNormalization()(z)
    ## Layer 15
    z = Conv2D(256, (3,3), strides=(1,1), activation='relu', padding='same', name='conv_12')(z)
    z = BatchNormalization()(z)

    # Fusion Layer
    y = RepeatLayer(32)(y)
    f = Concatenate()([y, z])
    f = FusionLayer(256)(f)
    f = Activation("relu")(f)

    # Colorization network

    ## Layer 17
    c = Conv2D(128, (3,3), strides=(1,1), activation='relu', padding='same', name='conv_13')(f)
    c = BatchNormalization()(c)
    ## Layer 18
    c = UpSampling2D(size=(2, 2))(c)
    ## Layer 19
    c = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same', name='conv_14')(c)
    c = BatchNormalization()(c)
    ## Layer 20
    c = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same', name='conv_15')(c)
    c = BatchNormalization()(c)
    ## Layer 21
    c = UpSampling2D(size=(2, 2))(c)
    ## Layer 22
    c = Conv2D(32, (3,3), strides=(1,1), activation='relu', padding='same', name='conv_16')(c)
    c = BatchNormalization()(c)
    ## Output Layer
    c = Conv2D(2, (3,3), strides=(1,1), activation='sigmoid', padding='same', name='conv_17')(c)
    output = UpSampling2D(size=(2, 2))(c)

    self.model = Model(input_image, output)

    if is_model_saved(self.name) and self.load_saved:
      self.model = load_model(self.model, self.name)
      print("Model Loaded!")



    self.model.compile(optimizer=self.optimizer, loss=self.loss)


class FusionLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(FusionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weights variables for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[-1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        self.bias = self.add_weight(name='bias', 
                                      shape=(256,),
                                      initializer='uniform',
                                      trainable=True)
        super(FusionLayer, self).build(input_shape)

    def call(self, x):
        self.bias = K.expand_dims(self.bias, axis=1)
        self.bias = K.repeat_elements(self.bias, rep=32, axis=1)
        self.bias = K.expand_dims(self.bias, axis=2)
        self.bias = K.repeat_elements(self.bias, rep=32, axis=2)
        self.bias = K.permute_dimensions(self.bias, (2,1,0))
        fusion = K.dot(x, self.kernel) + self.bias
        return fusion

    def compute_output_shape(self, input_shape):
        return (None, 32,32,256)


class RepeatLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(RepeatLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(RepeatLayer, self).build(input_shape)

    def call(self, x):
        y = K.expand_dims(x, axis=1)
        y = K.repeat_elements(y, rep=32, axis=1)
        y = K.expand_dims(y, axis=2)
        y = K.repeat_elements(y, rep=32, axis=2)
        return y

    def compute_output_shape(self, input_shape):
        return (None, 32,32,256)

