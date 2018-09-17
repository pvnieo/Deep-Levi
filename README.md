# Deep-Levi
Implementation of some automatic colorization models using deep neural network:

1. Implementation of the regression-based model provided in: "Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors for Automatic Image Colorization with Simultaneous Classification" [Link to the original paper](http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/en/)
2.  Implementation of a classification-based model  inspired in part by Zhang et al. in "Colorful Image Colorization"  [[Link to the original paper](https://github.com/richzhang/colorization)] and R.Dah in [here](http://tinyclouds.org/colorize)
3.  Implementation of a regression-based model inspired from [this Medium blog article](https://medium.freecodecamp.org/colorize-b-w-photos-with-a-100-line-neural-network-53d9b4449f8d)

## Project report
You can consult the project report [here](https://github.com/pvnieo/Deep-Levi/blob/master/%C3%89tude%20comparative%20d%E2%80%99architecture%20de%20colorisation%20automatique.pdf) (in French)

## Installation

This project runs on python >= 3.6, use pip to install dependencies:

```
pip3 install -r requirements.txt
```

## Usage

Use the `main.py` script to choose the model to train and the parameters to use

```
usage: main.py [-h] [-m {naive,reg,classif}] [-d DATA_DIR] [-e EPOCHS]
               [-b BATCH_SIZE] [-tc TO_COLOR] [--no-load] [--no-save]
               [--no-train] [--early]

An implementation of multiple approachs to automatically colorize grey-scale
images

optional arguments:
  -h, --help            show this help message and exit
  -m {naive,reg,classif}, --model {naive,reg,classif}
                        Colorization model
  -d DATA_DIR, --data-dir DATA_DIR
                        Directory where data is
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs to train for
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size to use
  -tc TO_COLOR, --to-color TO_COLOR
                        Number of samples to be colored
  --no-load             Disable loading saved model
  --no-save             Disable saving the new model
  --no-train            Disable training the model
  --early               Enable early stopping
```

Use the `levi.py` script to color images using a model

```
usage: levi.py -h [-d DATA_DIR]

An implementation of multiple approachs to automatically colorize grey-scale

images

optional arguments:

  -h, --help            show this help message and exit

  -m {naive,reg,classif}, --model {naive,reg,classif}

                        Colorization model

  -d DATA_DIR, --data-dir DATA_DIR

                        Directory where the images to be colored is

```

## Note

This project is under development. The final objective being to create a model that colors mangas by learning on the corresponding anime, this project will be updated regularly.
