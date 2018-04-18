import argparse
# project
from models.naive import Naive
from models.regression import Regression
from models.classification import Classification
import utils
import numpy as np
from skimage.color import rgb2lab, lab2rgb
import os
from skimage.io import imsave

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="An implementation of multiple approachs to automatically colorize grey-scale images")
  parser.add_argument("-m", "--model", help="Colorization model", type=str, default="naive", choices=["naive", "reg", "classif"])
  parser.add_argument('-d', '--data-dir',   required=False, default="levi_test",help='Directory where data is')

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


  files = os.listdir(to_color_dir)
  if not os.path.exists(TRAIN_DESTINATION):
      os.makedirs(TRAIN_DESTINATION)
  for i, file in enumerate(files):
    to_predict = np.array([img_to_array(load_img(to_color_dir + "/" + file)) ], dtype=float) * 1/255
    output = np.zeros(to_predict.shape)
    to_predict = rgb2lab(to_predict)
    to_predict = [to_predict]
    output[:,:,0] = to_predict[0][:,:,0]
    to_predict = utils.set_data_input(to_predict[:,:,:,0], selected_model.input_type)
    predicted = selected_model.predict(to_predict)
    output[:,:,1:] = from_output_to_image(predicted, selected_model.input_type)
    imsave("{}/predicted_{}.png".format(selected_model.name, file), lab2rgb(output))




