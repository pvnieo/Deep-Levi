from random import shuffle
import os

TRAIN_DESTINATION = 'train/1/'
TEST_DESTINATION = 'test/1/'
VALID_DASTINATION = 'validation/1/'

if not os.path.exists(TRAIN_DESTINATION):
      os.makedirs(TRAIN_DESTINATION)
if not os.path.exists(VALID_DASTINATION):
      os.makedirs(VALID_DASTINATION)
if not os.path.exists(TEST_DESTINATION):
      os.makedirs(TEST_DESTINATION)

files = os.listdir('.')
print(files)
shuffle(files)
files.remove('test_train_creator.py')

files.remove(TRAIN_DESTINATION[:-3])
files.remove(TEST_DESTINATION[:-3])
files.remove(VALID_DASTINATION[:-3])

train = files[:int(len(files)*0.8)]
train_files = train[:int(len(train)*0.8)]
valid_files = train[int(len(train)*0.8):]
test_files = files[int(len(files)*0.8):]

for filename in train_files:
  os.rename(filename, TRAIN_DESTINATION + filename)

for filename in test_files:
  os.rename(filename, TEST_DESTINATION  + filename)

for filename in valid_files:
  os.rename(filename, VALID_DASTINATION + filename)



