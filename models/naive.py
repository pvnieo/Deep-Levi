#project
from utils import is_model_saved, load_model
# 3p
from keras.models import Sequential
from keras.layers import Conv2D, UpSampling2D
from keras.layers import Activation, InputLayer
from keras.callbacks import TensorBoard
from keras import optimizers
from keras import losses


class Naive:

  def __init__(self, load, optimizer='adam', loss=losses.mean_squared_error):
    self.load_saved = load
    self.optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    self.loss = loss
    self.callbacks = []
    self.name = "naive"
    self.input_type = "reg"
    self.model = Sequential()

    # Define model

    self.model = Sequential()
    self.model.add(InputLayer(input_shape=(256, 256, 1)))
    self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
    self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
    self.model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    self.model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
    self.model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    self.model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    self.model.add(UpSampling2D((2, 2)))
    self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    self.model.add(UpSampling2D((2, 2)))
    self.model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    self.model.add(Conv2D(2, (3, 3), activation='sigmoid', padding='same'))
    self.model.add(UpSampling2D((2, 2)))

    if is_model_saved(self.name) and self.load_saved:
      self.model = load_model(self.model, self.name)
      print("Model Loaded!")

    self.model.compile(optimizer=self.optimizer, loss=self.loss)




