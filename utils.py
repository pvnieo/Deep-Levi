import os, os.path
import numpy as np
import tensorflow as tf
from tensorflow import convert_to_tensor as ctt
import numpy as np
from keras import losses
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler, ReduceLROnPlateau
from keras.utils import to_categorical as tc
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave

# ==================> Saving and Loading models <==================
def is_model_saved(name):
  weights = "saved_models/" + name + ".h5"
  return os.path.isfile(weights)

def load_model(model, name):
  weights = "saved_models/" + name + ".h5"
  model = model
  model.load_weights(weights)
  return model

def save_model(model):
    root = "saved_models/" + model.name
    model.model.save_weights(root + ".h5")

# ==================> Data processing <==================
def load_data(directory):
  PATH_TO_TRAIN = directory + "/train/"
  PATH_TO_TEST = directory + "/test/"
  # Load images (train + test)
  train_set = np.array([img_to_array(load_img(PATH_TO_TRAIN  + x)) for x in os.listdir(PATH_TO_TRAIN)], dtype=float)
  test_set = np.array([img_to_array(load_img(PATH_TO_TEST  + x)) for x in os.listdir(PATH_TO_TEST)], dtype=float)
  print("7ad hna mezyane", train_set.shape)
  # Data normalisation
  train_set *= 1.0/255
  test_set *= 1.0/255
  return train_set, test_set

def steps(train_dir, valid_dir, test_dir, batch_size):
  len_train = len([name for name in os.listdir(train_dir+"/1") ])
  len_valid = len([name for name in os.listdir(valid_dir+"/1") ])
  len_test = len([name for name in os.listdir(test_dir)])
  return (max(1,len_train // batch_size), max(1, len_valid // batch_size), max(1, len_test // batch_size))

def set_data_input(data, input_type):
  if input_type == "reg": # If reg, return the L channel
    data = data
    return data.reshape(data.shape+(1,))
  elif input_type == "cls": # If cls, retrun the L channel duplicated 3 times
    data = np.expand_dims(data, axis=-1)
    data = np.repeat(data, axis=-1, repeats=3)
    return data

def set_data_output(data, input_type):
  data = normalize_lab(data)
  if input_type == "reg": # if reg, return ab channels normalized
    return data
  elif input_type == "cls": # if cls, return ab classified version and concatenated
    def f(x): return int(x/4)
    f = np.vectorize(f)
    data *= 255
    data_a = data[:,:,:,0]
    data_a = f(data_a)
    data_b = data[:,:,:,1]
    data_b = f(data_b)
    data_a = tc(data_a, num_classes=64)
    data_b = tc(data_b, num_classes=64)
    data = np.concatenate((data_a, data_b),axis=1)
    return data

def from_output_to_image(data, input_type):
  if input_type == "reg":
    data = denormalize_lab(data)
    return data
  elif input_type == "cls":
    a_channel_classes = data[:224,:,:]
    b_channel_classes = data[224:,:,:]
    a_channel_bin = np.apply_along_axis(np.argmax, -1, a_channel_classes)
    b_channel_bin = np.apply_along_axis(np.argmax, -1, b_channel_classes)
    a_channel = a_channel_bin / 63
    b_channel =  b_channel_bin / 63
    output = np.zeros(a_channel.shape + (2,))
    output[:,:,0] = a_channel
    output[:,:,1] = b_channel
    output = denormalize_lab(output)
    return output

def from_input_to_image(data, input_type):
  if input_type == "reg":
    data = get_ride_of_additionnal_dim(data)
    return data
  elif input_type == "cls":
    data = data[:,:,0]
    return data

def train_generator(train_dir, target_size, batch_size, input_type):
  # Data augumentation generator
  datagen = ImageDataGenerator(
          shear_range=0.2,
          zoom_range=0.2,
          width_shift_range=0.2,
          height_shift_range=0.2,
          rotation_range=20,
          horizontal_flip=True, 
          rescale=1./255)
  for batch in datagen.flow_from_directory(train_dir, target_size=target_size, class_mode=None, batch_size=batch_size):
      lab_channel = rgb2lab(batch)
      l_channel = lab_channel[:,:,:,0]
      ab_channel = lab_channel[:,:,:,1:]
      X_batch = set_data_input(l_channel, input_type)
      Y_batch = set_data_output(ab_channel, input_type)
      yield (X_batch, Y_batch)

def valid_generator(train_dir, target_size, batch_size, input_type):
  datagen = ImageDataGenerator(rescale=1./255)
  print("taille dyal retourn gen", train_dir)
  for batch in datagen.flow_from_directory(train_dir, target_size=target_size, class_mode=None, batch_size=batch_size, shuffle=False):
      lab_channel = rgb2lab(batch)
      l_channel = lab_channel[:,:,:,0]
      ab_channel = lab_channel[:,:,:,1:]
      X_batch = set_data_input(l_channel, input_type)
      Y_batch = set_data_output(ab_channel, input_type)
      yield (X_batch, Y_batch)

def get_test_data(test_set, input_type):
  l_channel = rgb2lab(test_set)[:,:,:,0]
  ab_channel = rgb2lab(test_set)[:,:,:,1:]
  X_batch = set_data_input(l_channel, input_type)
  Y_batch = set_data_output(ab_channel, input_type)
  return X_batch, Y_batch

def save_sample(sample, input_type, directory, prefix, i, epochs, batch_size):
  gray_scale = from_input_to_image(sample[0], input_type)
  if input_type == "reg":
    gt = np.zeros((256, 256, 3))
    pred = np.zeros((256, 256, 3))
  elif input_type == "cls":
    gt = np.zeros((224, 224, 3))
    pred = np.zeros((224, 224, 3))
  ab_gt = from_output_to_image(sample[1], input_type)
  ab_pred = from_output_to_image(sample[2], input_type)
  gt[:,:,0] = gray_scale
  gt[:,:,1:] = ab_gt
  pred[:,:,0] = gray_scale
  pred[:,:,1] = np.clip(ab_pred[:,:,0], -127, 128)
  pred[:,:,2] = np.clip(ab_pred[:,:,1], -128, 127)
  bw = gray_scale / 100
  imsave("{}/{}bw{}_{}e_{}bz.png".format(directory, prefix, str(i), epochs, batch_size), bw)
  imsave("{}/{}gt{}_{}e_{}bz.png".format(directory, prefix, str(i), epochs, batch_size), lab2rgb(gt))
  imsave("{}/{}pred{}_{}e_{}bz.png".format(directory, prefix, str(i), epochs, batch_size), lab2rgb(pred))


def save_colored_samples(model, test_dir, steps, to_color, epochs, batch_size):
  print("hahiya bent l9a7ba", test_dir[:-2  ])
  output = model.model.predict_generator(valid_generator(test_dir[:-2], model.target_size, batch_size, model.input_type), 
                                         steps=steps,
                                         verbose=1)

  test_set = np.array([img_to_array(load_img(test_dir + "/" + x)) for x in os.listdir(test_dir)], dtype=float) * 1/255
  color_me = rgb2lab(test_set)[:,:,:,0]
  color_me = set_data_input(color_me, model.input_type)

  ground_truth = rgb2lab(test_set)[:,:,:,1:]
  ground_truth = set_data_output(ground_truth, model.input_type)


  # Take the N first good and bad colorization
  zipped = list(zip(color_me, ground_truth, output)) # [(bw, gt, output)]

  to_be_saved = sorted(zipped, key=lambda x: np.sum(tf.keras.backend.eval(model.model.loss(ctt(x[2], dtype="float32"), ctt(x[1], dtype="float32")))))[:to_color]
  to_be_saved_bad = sorted(zipped, key=lambda x: np.sum(tf.keras.backend.eval(model.model.loss(ctt(x[2], dtype="float32"), ctt(x[1], dtype="float32")))))[-to_color:]

  # Save Output colorizations
  # Check if directory exists
  directory = 'color_me/' + model.name + '/{}e_{}bz'.format(epochs, batch_size)
  if not os.path.exists(directory):
      os.makedirs(directory)
  # Save!
  for i in range(len(to_be_saved)):
    save_sample(to_be_saved[i], model.input_type, directory, "G", i, epochs, batch_size)
  for i in range(len(to_be_saved_bad)):
    save_sample(to_be_saved_bad[i], model.input_type, directory, "B", i, epochs, batch_size)

def get_ride_of_additionnal_dim(l):
  l = np.array(l)
  return np.squeeze(l, axis=len(l.shape)-1)

def normalize_lab(l):
  cur = np.zeros(l.shape)
  cur[:,:,:,0] = l[:,:,:,0] + 127
  cur[:,:,:,1] = l[:,:,:,1] + 128
  return cur / 255

def denormalize_lab(l):
  l *=  255
  cur = np.zeros(l.shape)
  cur[:,:,0] = l[:,:,0] - 127
  cur[:,:,1] = l[:,:,1] - 128
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

def reducelr_callback():
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                patience=5, min_lr=0.001)
  return reduce_lr

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
