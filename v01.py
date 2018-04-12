# floyd run --mode jupyter --env keras --data schlodinger/datasets/levi-data/1:my-data
# PP
import os
import random
import numpy as np
from math import floor
# 3P
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, UpSampling2D
from keras.layers import Activation, InputLayer
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

PATH_TO_TRAIN = "datasets/mit_opencountry_dataset/train1/"
PATH_TO_TEST = "datasets/mit_opencountry_dataset/test1/"
BATCH_SIZE = 10
EPOCHS = 12

# Load images (train + test)
train_set = np.array([img_to_array(load_img(PATH_TO_TRAIN  + x)) for x in os.listdir(PATH_TO_TRAIN)], dtype=float)
test_set = np.array([img_to_array(load_img(PATH_TO_TEST  + x)) for x in os.listdir(PATH_TO_TEST)], dtype=float)

#set 8-bit images
train_set = 1.0/255 * train_set
test_set = 1.0/255 * test_set

# Definiton of NN
model = Sequential()
model.add(InputLayer(input_shape=(256, 256, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))
model.compile(optimizer='rmsprop', loss='mse')

# Data augumentation generator
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)

# Generate training data
def image_a_b_gen(batch_size):
    for batch in datagen.flow(train_set, batch_size=batch_size):
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)

# Train model      
tensorboard = TensorBoard(log_dir="output/beta_run")
model.fit_generator(image_a_b_gen(BATCH_SIZE), callbacks=[tensorboard], epochs=EPOCHS, steps_per_epoch=1)

# Save model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")

# Testing model
Xtest = rgb2lab(test_set)[:,:,:,0]
Xtest = Xtest.reshape(Xtest.shape+(1,))
Ytest = rgb2lab(test_set)[:,:,:,1:]
Ytest = Ytest / 128
# Printing result
print(model.evaluate(Xtest, Ytest, batch_size=BATCH_SIZE))

# Let's color some images
TO_COLOR = 10
color_me = rgb2lab(test_set)[0:TO_COLOR,:,:,0]
color_me = color_me.reshape(color_me.shape+(1,))

output = model.predict(color_me)
output = output * 128

# Save Output colorizations
for i in range(len(output)):
  cur = np.zeros((256, 256, 3))
  cur[:,:,0] = color_me[i][:,:,0]
  st = cur
  cur[:,:,1:] = output[i]
  imsave("result/img_"+str(i)+".png", lab2rgb(cur))
  imsave("result/img_bw"+str(i)+".png", rgb2gray(lab2rgb(cur)))
