import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import h5py
import math


def meanPred(y_true, y_pred):
    return K.mean(y_pred)


def loadDataset():
    train_dataset = h5py.File('datasets/train_happy.h5')
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('datasets/test_happy.h5')
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def plotAccuracy(val):
    plt.plot(val)
    #plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.grid()
    #plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def plotLoss(val):
    plt.plot(val)
    #plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.grid()
    #plt.legend(['train', 'test'], loc='upper left')
    plt.show()
