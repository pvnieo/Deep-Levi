from math import floor
# project
from models.naive import Naive
from models.regression import Regression
import utils

# 3P
import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="An implementation of multiple approachs to automatically colorize grey-scale images")
  parser.add_argument("-m", "--model", help="Colorization model", type=str, default="naive", choices=["naive", "reg"])
  parser.add_argument('-d', '--data-dir',   required=False, default="datasets/test",help='Directory where data is')
  parser.add_argument('-e', '--epochs',     required=False,default=10,type=int,help='Number of epochs to train for')
  parser.add_argument('-b', '--batch-size', required=False,type=int,default=32,help='Batch size to use')
  parser.add_argument("-tc", "--to-color", help="Number of samples to be colored", type=int, default=10)

  args = parser.parse_args()

  # Hyperparameters
  BATCH_SIZE = args.batch_size
  EPOCHS = args.epochs

  # Load training and testing data
  DATA_DIR = args.data_dir
  train_set, test_set = utils.load_data(DATA_DIR)
  alpha = 0.8
  TRAIN_VALID_SPLIT = floor(len(train_set) * alpha)
  STEPS_PER_EPOCHS = floor(len(train_set) * alpha / BATCH_SIZE)
  STEPS_PER_EPOCHS_VALID = max(floor(len(train_set) * (1 - alpha) / BATCH_SIZE), 1)
  print()
  print('MODEL:           ',args.model)
  print('DATA_DIR:        ',DATA_DIR)
  print('EPOCHS:          ',EPOCHS)
  print('BATCH_SIZE:      ',BATCH_SIZE, '\n')

  if args.model == "naive":
    selected_model = Naive()
  elif args.model == "reg":
    selected_model = Regression()

  print(selected_model.model.summary())

  # Setting Callbacks
  callbacks = [utils.checkpoint_callback(selected_model.name), 
               utils.tensorboard_callback(selected_model.name)]

  # Train model
  history = selected_model.model.fit_generator(generator=utils.train_generator(train_set, BATCH_SIZE, TRAIN_VALID_SPLIT), 
                                      steps_per_epoch=STEPS_PER_EPOCHS,
                                      callbacks=callbacks, 
                                      validation_data  = utils.valid_generator(train_set, BATCH_SIZE, TRAIN_VALID_SPLIT), 
                                      validation_steps = STEPS_PER_EPOCHS_VALID, 
                                      epochs=EPOCHS)
  
  # print(history.history)

  # Save model
  utils.save_model(selected_model)

  # Get test data
  Xtest, Ytest = utils.get_test_data(test_set)

  # Printing generalisation loss
  print("====> Generalisation loss: ", selected_model.model.evaluate(Xtest, Ytest, batch_size=BATCH_SIZE))

  # Let's color some images
  TO_COLOR = args.to_color
  utils.save_colored_samples(selected_model, test_set, TO_COLOR, EPOCHS, BATCH_SIZE)
