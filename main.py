from math import floor
# project
from models.naive import Naive
from models.regression import Regression
from models.classification import Classification
import utils

# 3P
import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="An implementation of multiple approachs to automatically colorize grey-scale images")
  parser.add_argument("-m", "--model", help="Colorization model", type=str, default="naive", choices=["naive", "reg", "classif"])
  parser.add_argument('-d', '--data-dir',   required=False, default="datasets/test",help='Directory where data is')
  parser.add_argument('-e', '--epochs',     required=False,default=10,type=int,help='Number of epochs to train for')
  parser.add_argument('-b', '--batch-size', required=False,type=int,default=32,help='Batch size to use')
  parser.add_argument("-tc", "--to-color", help="Number of samples to be colored", type=int, default=10)
  parser.add_argument("--no-load", help="Disable loading saved model", action="store_true")
  parser.add_argument("--no-save", help="Disable saving the new model", action="store_true")
  parser.add_argument("--no-train", help="Disable training the model", action="store_true")
  parser.add_argument("--early", help="Enable early stopping", action="store_true")
  args = parser.parse_args()

  # Hyperparameters
  BATCH_SIZE = args.batch_size
  EPOCHS = args.epochs
  LOAD = not args.no_load
  SAVE = not args.no_save
  TRAIN = not args.no_train

  # Load training and testing data
  DATA_DIR = args.data_dir
  DATA_TRAIN_DIR = DATA_DIR + "/train"
  DATA_VALID_DIR = DATA_DIR + "/validation"
  DATA_TEST_DIR = DATA_DIR + "/test/1"
  # train_set, test_set = utils.load_data(DATA_DIR)

  
  STEPS_PER_EPOCHS, STEPS_PER_EPOCHS_VALID, STEPS_PER_EPOCHS_TEST = utils.steps(DATA_TRAIN_DIR, DATA_VALID_DIR, DATA_TEST_DIR, BATCH_SIZE)

  print()
  print('MODEL:           ',args.model)
  print('DATA_DIR:        ',DATA_DIR)
  print('EPOCHS:          ',EPOCHS)
  print('BATCH_SIZE:      ',BATCH_SIZE, '\n')

  if args.model == "naive":
    selected_model = Naive(LOAD)
  elif args.model == "reg":
    selected_model = Regression(LOAD)
  elif args.model == "classif":
    selected_model = Classification(LOAD)

  print(selected_model.model.summary())

  # Setting Callbacks
  callbacks = [utils.checkpoint_callback(selected_model.name), 
               utils.tensorboard_callback(selected_model.name), 
               utils.reducelr_callback()]

  if args.early:
    callbacks.append(utils.earlystopping_callback())

  if TRAIN:
    # Train model
    history = selected_model.model.fit_generator(generator=utils.train_generator(DATA_TRAIN_DIR, selected_model.target_size, 
                                                                                  BATCH_SIZE, selected_model.input_type), 
                                                 steps_per_epoch=STEPS_PER_EPOCHS, 
                                                 callbacks=callbacks, 
                                                 validation_data  = utils.valid_generator(DATA_VALID_DIR, selected_model.target_size, 
                                                                                          BATCH_SIZE, selected_model.input_type), 
                                                 validation_steps = STEPS_PER_EPOCHS_VALID, 
                                                 epochs=EPOCHS)
  
  # print(history.history)

  # Save model
  # if LOAD and SAVE: # We don't save the new weights if we didn't load
  #   utils.save_model(selected_model)

  # Get test data
  #Xtest, Ytest = utils.get_test_data(test_set, selected_model.input_type)

  # Printing generalisation loss
  # print("====> Generalisation loss: ", selected_model.model.evaluate(Xtest, Ytest, batch_size=BATCH_SIZE))

  # Let's color some images
  TO_COLOR = args.to_color
  utils.save_colored_samples(selected_model, DATA_TEST_DIR, STEPS_PER_EPOCHS_TEST, TO_COLOR, EPOCHS, BATCH_SIZE)
