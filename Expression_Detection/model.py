import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import keras 

from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from matplotlib.pyplot import imshow
from expression_detection_utils import *

K.set_image_data_format('channels_last')


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = loadDataset()

# Normalize image vectors 
X_train = X_train_orig/255 
X_test = X_test_orig/255

# Reshape 
Y_train = Y_train_orig.T 
Y_test = Y_test_orig.T

print("number of training examples: "+str(X_train.shape[0]))
print("number of test examples: "+str(X_test.shape[0]))
print("X_train shape: "+str(X_train.shape))
print("Y_train shape: "+str(Y_train.shape))
print("X_test shape: "+str(X_test.shape))
print("Y_test shape: "+str(Y_test.shape))


def emotionModel(input_shape):
    """
    Implementation of emotion detector model
    :param input_shape: shape of the input images (height, width,
    channels) as a tuple. Batch is not included as an input dimension
    :return: model for the emotion detection
    """

    # Define input placeholder as a tensor having shape 'input_shape'
    X_input = Input(input_shape)

    # Zero Padding: pads the border of X_input with zeros
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> ReLU block applied to X
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN (to a vector) + FC
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='emotionModel')

    return model


emotionModel = emotionModel(X_train.shape[1:])
emotionModel.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
history = emotionModel.fit(X_train, Y_train, epochs=30, batch_size=50)
print(history.history.keys())
plotAccuracy(history.history['acc'])
plotLoss(history.history['loss'])


predictions = emotionModel.evaluate(X_test, Y_test, batch_size=32, verbose=1,
                                    sample_weight=None)

print()
print("Loss = " + str(predictions[0]))
print("Test Accuracy = " + str(predictions[1]))

# Inference
# img_path = 'my_image.jpg'
# img = image.load_img(img_path, target_size=(64, 64))
# imshow(img)

# x = image.img_to_array(img)
# x = np.expand_dims(x, axis = 0)
# x = preprocess_input(x)










