import argparse
# project
from models.naive import Naive
from models.regression import Regression
from models.classification import Classification
import utils
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
from os.path import isfile, join
from skimage.io import imsave

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="An implementation of multiple approachs to automatically colorize grey-scale images")
  parser.add_argument("-m", "--model", help="Colorization model", type=str, default="naive", choices=["naive", "reg", "classif"])
  parser.add_argument('-d', '--data-dir',   required=False, default="levi_test",help='Directory where data is')
  args = parser.parse_args()

  LOAD = True
  to_color_dir = args.data_dir


  if args.model == "naive":
    selected_model = Naive(LOAD)
  elif args.model == "reg":
    selected_model = Regression(LOAD)
  elif args.model == "classif":
    selected_model = Classification(LOAD)

  if not os.path.exists(to_color_dir + "/{}".format(selected_model.name)):
      os.makedirs(to_color_dir + "/{}".format(selected_model.name))


  files =  [f for f in os.listdir(to_color_dir) if isfile(join(to_color_dir, f))]
  
  for i, file in enumerate(files):
    to_predict = np.array(img_to_array(load_img(to_color_dir + "/" + file)), dtype=float) 
    to_predict = to_predict * 1/255
    output = np.zeros(to_predict.shape)
    to_predict = rgb2lab(to_predict)
    output[:,:,0] = to_predict[:,:,0]
    to_predict = utils.set_data_input(to_predict[:,:,0], selected_model.input_type)
    to_predict = np.expand_dims(to_predict, axis=0)
    predicted = selected_model.model.predict(to_predict)
    ab = utils.from_output_to_image(predicted, selected_model.input_type)[0]
    output[:,:,1] = np.clip(ab[:,:,0], -127, 128)
    output[:,:,2] = np.clip(ab[:,:,1], -128, 127)
    imsave("{}/{}/pred_{}.png".format(to_color_dir, selected_model.name, file), lab2rgb(output))




