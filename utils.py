import os, os.path
from time import time
import numpy as np
import tensorflow as tf
import numpy as np
from keras import losses
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.utils import to_categorical as tc
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave

# ==================> Saving and Loading models <==================
def is_model_saved(name):
  weights = "saved_models/" + name + ".h5"
  structure = "saved_models/" + name + ".json"
  return os.path.isfile(weights) and os.path.isfile(structure)

def load_model(name):
  weights = "saved_models/" + name + ".h5"
  structure = "saved_models/" + name + ".json"
  with open(structure,'r') as f:
      output = f.read()
  model = model_from_json(output)
  model.load_weights(weights)
  return model

def save_model(model):
    root = "saved_models/" + model.name
    model_json = model.model.to_json()
    with open(root + ".json", "w") as json_file:
        json_file.write(model_json)
    model.model.save_weights(root + ".h5")

# ==================> Data processing <==================
def load_data(directory):
  PATH_TO_TRAIN = directory + "/train/"
  PATH_TO_TEST = directory + "/test/"
  # Load images (train + test)
  train_set = np.array([img_to_array(load_img(PATH_TO_TRAIN  + x)) for x in os.listdir(PATH_TO_TRAIN)], dtype=float)
  test_set = np.array([img_to_array(load_img(PATH_TO_TEST  + x)) for x in os.listdir(PATH_TO_TEST)], dtype=float)
  # Data normalisation
  train_set = 1.0/255 * train_set
  test_set = 1.0/255 * test_set
  return train_set, test_set

def train_generator(train_set, batch_size, split, input_type):
  # Data augumentation generator
  datagen = ImageDataGenerator(
          shear_range=0.2,
          zoom_range=0.2,
          rotation_range=20,
          horizontal_flip=True)
  for batch in datagen.flow(train_set[:split], batch_size=batch_size):
      lab_batch = rgb2lab(batch)
      X_batch = lab_batch[:,:,:,0]
      Y_batch = normalize_lab(lab_batch[:,:,:,1:])
      if input_type = "cls":
        X_batch = np.expand_dims(X_batch, axis=-1)
        X_batch = np.repeat(X_batch, axis=-1, repeats=3)
        Y_batch_1 = tc(Y_batch[:,:,:,0], num_classes=64)
        Y_batch_2 = tc(Y_batch[:,:,:,1], num_classes=64)
        Y_batch = np.concatenate((Y_batch_1, Y_batch_2),axis=0)
      yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)

def valid_generator(train_set, batch_size, split, input_type):
  datagen = ImageDataGenerator()
  for batch in datagen.flow(train_set[split:], batch_size=batch_size):
      lab_batch = rgb2lab(batch)
      X_batch = lab_batch[:,:,:,0]
      Y_batch = normalize_lab(lab_batch[:,:,:,1:])
      if input_type = "cls":
        X_batch = np.expand_dims(X_batch, axis=-1)
        X_batch = np.repeat(X_batch, axis=-1, repeats=3)
        Y_batch_1 = tc(Y_batch[:,:,:,0], num_classes=64)
        Y_batch_2 = tc(Y_batch[:,:,:,1], num_classes=64)
        Y_batch = np.concatenate((Y_batch_1, Y_batch_2),axis=0)
      yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)

def get_test_data(test_set):
  Xtest = rgb2lab(test_set)[:,:,:,0]
  Ytest = rgb2lab(test_set)[:,:,:,1:]
  Ytest = normalize_lab(Ytest)
  if input_type = "cls":
    Xtest = np.expand_dims(Xtest, axis=-1)
    Xtest = np.repeat(Xtest, axis=-1, repeats=3)
    Y_batch_1 = tc(Ytest[:,:,:,0], num_classes=64)
    Y_batch_2 = tc(Ytest[:,:,:,1], num_classes=64)
    Ytest = np.concatenate((Y_batch_1, Y_batch_2),axis=0)
  Xtest = Xtest.reshape(Xtest.shape+(1,))
  return Xtest, Ytest

def save_colored_samples(model, test_set, to_color, epochs, batch_size):
  color_me = rgb2lab(test_set)[:,:,:,0]
  color_me = color_me.reshape(color_me.shape+(1,))

  ground_truth = rgb2lab(test_set)[:,:,:,1:]
  ground_truth = ground_truth.reshape(ground_truth.shape+(1,))

  output = model.model.predict(color_me)
  # output = output * 128
  output = denormalize_lab(output)
  output = output.reshape(output.shape+(1,))
  # Take the N first good colorization
  zipped = list(zip(color_me, ground_truth, output)) # [(bw, gt, output)]
  to_be_saved = sorted(zipped, key=lambda x: np.sum(tf.keras.backend.eval(losses.mean_squared_error(x[1],x[2]))))[:to_color]
  to_be_saved_bad = sorted(zipped, key=lambda x: np.sum(tf.keras.backend.eval(losses.mean_squared_error(x[1],x[2]))))[-to_color:]

  # Save Output colorizations
  # Check if directory exists
  directory = 'color_me/' + model.name + '/{}e_{}bz'.format(epochs, batch_size)
  if not os.path.exists(directory):
      os.makedirs(directory)
  # Save!
  for i in range(len(to_be_saved)):
    # Initialization
    cur = np.zeros((256, 256, 3))
    gt = np.zeros((256, 256, 3))
    # Get ride of add dim added for Keras input
    grey_scale = get_ride_of_additionnal_dim(to_be_saved[i][0])
    ab_gt = get_ride_of_additionnal_dim(to_be_saved[i][1])
    ab_predicted = to_be_saved[i][2]
    # Write images
    cur[:,:,0] = grey_scale
    gt[:,:,0] = grey_scale
    cur[:,:,1:2] = np.clip(ab_predicted[:,:,0], -127, 128)
    cur[:,:,2:] = np.clip(ab_predicted[:,:,1], -128, 127)
    gt[:,:,1:] = ab_gt
    bw = grey_scale / 100
    imsave("{}/Gbw{}_{}e_{}bz.png".format(directory, str(i), epochs, batch_size), bw)
    imsave("{}/Ggt{}_{}e_{}bz.png".format(directory, str(i), epochs, batch_size), lab2rgb(gt))
    imsave("{}/Gpred{}_{}e_{}bz.png".format(directory, str(i), epochs, batch_size), lab2rgb(cur))

  for i in range(len(to_be_saved_bad)):
    # Initialization
    cur = np.zeros((256, 256, 3))
    gt = np.zeros((256, 256, 3))
    # Get ride of add dim added for Keras input
    grey_scale = get_ride_of_additionnal_dim(to_be_saved_bad[i][0])
    ab_gt = get_ride_of_additionnal_dim(to_be_saved_bad[i][1])
    ab_predicted = to_be_saved_bad[i][2]
    # Write images
    cur[:,:,0] = grey_scale
    gt[:,:,0] = grey_scale
    cur[:,:,1:2] = np.clip(ab_predicted[:,:,0], -127, 127)
    cur[:,:,2:] = np.clip(ab_predicted[:,:,1], -127, 127)
    gt[:,:,1:] = ab_gt
    bw = grey_scale / 100
    imsave("{}/Bbw{}_{}e_{}bz.png".format(directory, str(i), epochs, batch_size), bw)
    imsave("{}/Bgt{}_{}e_{}bz.png".format(directory, str(i), epochs, batch_size), lab2rgb(gt))
    imsave("{}/Bpred{}_{}e_{}bz.png".format(directory, str(i), epochs, batch_size), lab2rgb(cur))

def get_ride_of_additionnal_dim(l):
  l = np.array(l)
  return np.squeeze(l, axis=len(l.shape)-1)
def normalize_lab(l):
  cur = np.zeros(l.shape)
  cur[:,:,:,0] = l[:,:,:,0] + 127
  cur[:,:,:,1] = l[:,:,:,1] + 128
  return cur / 255
def denormalize_lab(l):
  l = l * 255
  cur = np.zeros(l.shape)
  cur[:,:,:,0] = l[:,:,:,0] - 127
  cur[:,:,:,1] = l[:,:,:,1] - 128
  return cur



# ==================> Callbacks <==================
def checkpoint_callback(name):
  weights = "saved_models/" + name + ".h5"
  checkpoint = ModelCheckpoint(weights, 
                               monitor='val_loss', 
                               verbose=0, 
                               save_best_only=True, 
                               save_weights_only=True, 
                               mode='min', 
                               period=1)
  return checkpoint

def earlystopping_callback():
  early_stop = EarlyStopping(monitor='val_loss', 
                             min_delta=0.001, 
                             patience=3, 
                             mode='min', 
                             verbose=1)
  return early_stop

def tensorboard_callback(name):
  directory = 'logs/' + name
  if not os.path.exists(directory):
      os.makedirs(directory)

  ordre = len([log for log in os.listdir(directory)]) + 1
  tensorboard = TensorBoard(log_dir=directory + '/{}_{}'.format(name, ordre), 
                            histogram_freq=0, 
                            write_graph=True, 
                            write_images=False)
  return tensorboard

def learningratescheduler_callback():
  def triang1(e, lr):
    max_lr = ma
    min_lr = mi

  pass
  return LearningRateScheduler(triang1, verbose=0)

