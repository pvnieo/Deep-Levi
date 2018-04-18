#project
from utils import is_model_saved, load_model
# 3p
from keras import applications
from keras.utils.data_utils import get_file
from keras.models import Sequential, Model
from keras.layers import Conv2D, UpSampling2D, Dropout, Flatten, Dense, Input, Concatenate, MaxPooling2D, BatchNormalization
from keras.callbacks import TensorBoard
from keras.optimizers import SGD, Adadelta
from keras import losses

class Classification:

  def __init__(self, load, loss=losses.categorical_crossentropy):
    self.load_saved = load
    # Learning rate is changed to 0.001
    # self.optimizer = SGD(lr=5*1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    self.optimizer = Adadelta(lr=0.01, rho=0.95, epsilon=None, decay=0.0)
    self.loss = loss
    self.target_size = (224, 224)
    self.name = "classification"
    self.input_type = "cls"

    # Define model
    
    WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # build the VGG16 network
    input_shape = (224, 224, 3)
    img_input = Input(shape=input_shape)
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    vgg_model = Model(img_input, x, name='vgg16')
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                            WEIGHTS_PATH_NO_TOP,
                            cache_subdir='models',
                            file_hash='6d6bbae143d832006294945121d1f1fc')
    vgg_model.load_weights(weights_path)
    # print('VGG16 loaded!')
    # Freeze the 18 first layers
    for layer in vgg_model.layers[:18]:
      layer.trainable = False

    classifier = Conv2D(512, (3,3), activation='relu', padding='same')(vgg_model.outputs[-1])
    classifier = BatchNormalization()(classifier)
    classifier = UpSampling2D(size=(2, 2))(classifier)
    classifier = Conv2D(256, (3,3), activation='relu', padding='same')(classifier)
    classifier = BatchNormalization()(classifier)
    classifier = UpSampling2D(size=(2, 2))(classifier)
    classifier = Conv2D(256, (3,3), activation='relu', padding='same')(classifier)
    classifier = BatchNormalization()(classifier)
    classifier = UpSampling2D(size=(2, 2))(classifier)
    classifier = Conv2D(128, (3,3), activation='relu', padding='same')(classifier)
    classifier = BatchNormalization()(classifier)
    classifier = UpSampling2D(size=(2, 2))(classifier)

    # C*I*E*A* layer
    a_layer = Conv2D(64, (1,1), activation='softmax', padding='same')(classifier)
    a_layer = UpSampling2D(size=(2, 2))(a_layer)

    # C*I*E*B* layer
    b_layer = Conv2D(64, (1,1), activation='softmax', padding='same')(classifier)
    b_layer = UpSampling2D(size=(2, 2))(b_layer)

    # Concatenation and output layer
    output = Concatenate(axis=1)([a_layer, b_layer])

    self.model = Model(img_input, output)

    if is_model_saved(self.name) and self.load_saved:
      self.model = load_model(self.model, self.name)
      print("Model Loaded!")

    self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])




