from random import shuffle
import os

TRAIN_DESTINATION = 'train'
TEST_DESTINATION = 'test'

files = os.listdir('.')
shuffle(files)
# print(sorted(files))
files.remove('test_train_creator.py')
files.remove(TRAIN_DESTINATION)
files.remove(TEST_DESTINATION)
train_files = files[:int(2*len(files)/3)]
test_files = files[int(2*len(files)/3):]

for filename in train_files:
  os.rename(filename, TRAIN_DESTINATION + "/" + filename)

for filename in test_files:
  os.rename(filename, TEST_DESTINATION + "/" + filename)



