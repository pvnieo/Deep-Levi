import os, os.path
from time import time
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
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
    model_json = model.model.to_json()
    with open(model.name + ".json", "w") as json_file:
        json_file.write(model_json)
    model.model.save_weights(model.name + ".h5")

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

def image_a_b_gen(train_set, batch_size):
  # Data augumentation generator
  datagen = ImageDataGenerator(
          shear_range=0.2,
          zoom_range=0.2,
          rotation_range=20,
          horizontal_flip=True)
  for batch in datagen.flow(train_set, batch_size=batch_size):
      lab_batch = rgb2lab(batch)
      X_batch = lab_batch[:,:,:,0]
      Y_batch = lab_batch[:,:,:,1:] / 128
      yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)

def get_test_data(test_set):
  Xtest = rgb2lab(test_set)[:,:,:,0]
  Xtest = Xtest.reshape(Xtest.shape+(1,))
  Ytest = rgb2lab(test_set)[:,:,:,1:]
  Ytest = Ytest / 128
  return Xtest, Ytest

def save_colored_samples(model, test_set, to_color, name):
  color_me = rgb2lab(test_set)[:,:,:,0]
  color_me = color_me.reshape(color_me.shape+(1,))

  output = model.predict(color_me)
  output = output * 128

  # Save Output colorizations
  for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = color_me[i][:,:,0]
    st = cur
    cur[:,:,1:] = output[i]
    # TODO: Model saving image + create folder for each model, and each attempt
    imsave("color_me/{}_{}e_{}bz"+str(i)+".png", lab2rgb(cur))


# ==================> Callbacks <==================
def checkpoint_callback(name):
  weights = "saved_models/" + name + ".h5"
  checkpoint = ModelCheckpoint(weights, 
                               monitor='val_loss', 
                               verbose=1, 
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
  pass
