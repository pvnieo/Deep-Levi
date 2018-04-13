#project
from utils import is_model_saved, load_model
# 3p
from keras.models import Sequential
from keras.layers import Conv2D, UpSampling2D
from keras.layers import Activation, InputLayer
from keras.callbacks import TensorBoard


class Naive:

  def __init__(self, optimizer='rmsprop', loss='mse'):
    self.optimizer = optimizer
    self.loss = loss
    self.callbacks = []
    self.name = "naive"

    # Define model
    if is_model_saved(this.name):
      self.model = load_model(this.name)
    else:
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
      self.model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
      self.model.add(UpSampling2D((2, 2)))

    self.model.compile(optimizer=self.optimizer, loss=self.loss)




