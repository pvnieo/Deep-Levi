# project
from models import Naive
import utils

# 3P
import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="A deep learning approach to automatically colorize grey-scale images")
  parser.add_argument('-e', '--epochs',     required=False,default=10,type=int,help='Number of epochs to train for')
  parser.add_argument('-d', '--data-dir',   required=False, default="datasets/cvcl-mit/opencountry",help='Directory where data is')
  parser.add_argument('-b', '--batch-size', required=False,type=int,default=32,help='Batch size to use')
  parser.add_argument("-m", "--model", help="Colorization model", type=str, default="naive", choices=["naive"])
  parser.add_argument("-tc", "--to-color", help="Number of samples to be colored", type=int, default=10)

  args = parser.parse_args()

  # Hyperparameters
  BATCH_SIZE = args.batch_size
  EPOCHS = args.epochs

  # Load training and testing data
  DATA_DIR = args.data_dir
  train_set, test_set = utils.load_data(DATA_DIR)
  STEPS_PER_EPOCHS = floor(len(train_set) / BATCH_SIZE)

  if args.model == "naive":
    naive = Naive()

    # Setting Callbacks
    callbacks = [utils.checkpoint_callback(naive.name), utils.tensorboard_callback(naive.name)]

    # Train model
    naive.fit_generator(utils.image_a_b_gen(train_set, BATCH_SIZE), callbacks=callbacks, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCHS)

    # Save model
    utils.save_model(naive)

    # Get test data
    Xtest, Ytest = utils.get_test_data(test_set)

    # Printing generalisation loss
    print("====> Generalisation loss: " + model.evaluate(Xtest, Ytest, batch_size=BATCH_SIZE))

    # Let's color some images
    TO_COLOR = args.to_color
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
      imsave("results/img_"+str(i)+".png", lab2rgb(cur))
