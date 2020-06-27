# from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
from PIL import Image
from dnn_app_utils import *

import numpy as np
import time
import h5py
import matplotlib.pyplot as plt

np.random.seed(1)


def initialize_parameters(n_x_val, n_h_val, n_y_val):
    """
    :param n_x_val: size of input layer
    :param n_h_val: size of hidden layer
    :param n_y_val: size of output layer
    :return: initialized parameters
    W1: weight matrix of shape (n_h, n_x)
    b1: bias matrix of shape (n_h, 1)
    W2: weight matrix of shape (n_y, n_h)
    b2: bias matrix of shape (n_y, 1)
    """

    np.random.seed(1)

    W1 = np.random.randn(n_h_val, n_x_val) * 0.01
    b1 = np.zeros((n_h_val, 1))
    W2 = np.random.randn(n_y_val, n_h_val) * 0.01
    b2 = np.zeros((n_y_val, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2
                  }

    return parameters


def linear_forward(A_val, W, b):
    """
    :param A_val: activation from previous layer
    :param W: weight matrix of shape (size of current layer, size of previous layer)
    :param b: bias matrix of shape (size of current layer, 1)
    :return:
    Z: input of activation function of current input
    cache: tuple consisting of A, W, b for computing backward pass
    """

    Z = np.dot(W, A_val) + b
    cache = (A_val, W, b)

    return Z, cache


def linear_activation_forward(A_prev_val, W, b, activation):
    """
    :param A_prev_val: values of previous layer activations (size of previous layer, number of examples)
    :param W: weight matrix
    :param b: bias matrix
    :param activation: activation function to be used
    :return:
    A: activation values for the current layer
    cache: tuple consisting of linear cache and activation cache
    """

    if activation == "relu":
        Z, linear_cache = linear_forward(A_prev_val, W, b)
        A, activation_cache = relu(Z)

    elif activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev_val, W, b)
        A, activation_cache = sigmoid(Z)

    cache = (linear_cache, activation_cache)

    return A, cache


def compute_cost(AL_val, Y):
    """
    cost function implementation
    :param AL_val: probability vector corresponding to label prediction
    :param Y: true "label" vector
    :return:
    cost: cross entropy cost
    """

    m = Y.shape[1]

    cost = -(np.sum(Y * np.log(AL_val) + (1 - Y) * np.log(1 - AL_val))) / m

    cost = np.squeeze(cost)
    # print('cost: ', cost)

    return cost


def linear_backward(dZ_val, cache):
    """
    linear portion of back pass for a single layer
    :param dZ_val: gradient of cost function with respect to
    :param cache: tuple of (A_prev, W, b) coming from forward propagation
    :return:
    dA_prev: gradient of cost wrt activation function
    dW: gradient of cost wrt W
    db: gradient of cost wrt b
    """

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (np.dot(dZ_val, cache[0].T)) / m
    db = np.sum(dZ_val, axis=1, keepdims=True) / m
    dA_prev = np.dot(cache[1].T, dZ_val)

    return dA_prev, dW, db


def linear_activation_backward(dA_val, cache, activation):
    """

    :param dA_val: post activation gradient for current layer
    :param cache: (linear_cache, activation_cache) used for backward pass
    :param activation: activation function
    :return:
    dA_prev: gradient of cost wrt activation of previous layer
    dW: gradient of cost wrt W of current layer
    db: gradient of cost wrt b of current layer
    """

    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA_val, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA_val, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def update_parameters(old_parameters, grads, learning_rate):
    """
    :param old_parameters: weight and bias values
    :param grads: gradient values
    :param learning_rate: update factor
    :return:
    updated_parameters: incorporating the gradients and the learning rate
    """

    # L = len(parameters) // 2

    W1 = old_parameters["W1"]
    b1 = old_parameters["b1"]
    W2 = old_parameters["W2"]
    b2 = old_parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    updated_parameters = {"W1": W1,
                          "b1": b1,
                          "W2": W2,
                          "b2": b2}

    # for layer in range(L):
    #    parameters["W" + str(layer+1)] = parameters["W" + str(layer+1)] - learning_rate * grads["dW" + str(layer+1)]
    #    parameters["b" + str(layer+1)] = parameters["b" + str(layer+1)] - learning_rate * grads["db" + str(layer+1)]

    return updated_parameters


def two_layer_model(X, Y, layers_dims_val, learning_rate=0.0075, num_iterations=3000, print_cost=True):
    """

    :param X: input data (n_x, number of examples)
    :param Y: output data (true label)
    :param layers_dims_val: dimensions of (n_x, n_y, n_h)
    :param learning_rate: rate
    :param num_iterations: # of iterations
    :param print_cost: print cost after number of iterations
    :return: dictionary consisting of learned parameters
    """

    np.random.seed(1)
    grads = {}
    costs = []
    # m = X.shape[0]
    (n_x_val, n_h_val, n_y_val) = layers_dims_val

    parameters = initialize_parameters(n_x_val, n_h_val, n_y_val)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations):
        A1, cache1 = linear_activation_forward(X, parameters["W1"], parameters["b1"], "relu")
        A2, cache2 = linear_activation_forward(A1, parameters["W2"], parameters["b2"], "sigmoid")
        cost = compute_cost(A2, Y)

        dA2 = -(np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")

        grads["dW1"] = dW1
        grads["db1"] = db1
        grads["dW2"] = dW2
        grads["db2"] = db2

        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print('Cost after iteration {}: {}'.format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title('Learning rate= '+str(learning_rate))
    plt.grid()
    plt.show()
    return parameters


train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

train_x = train_x_flatten / 255
test_x = test_x_flatten / 255

n_x = 12288
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)

learned_parameters = two_layer_model(train_x, train_y, layers_dims, learning_rate=0.0075, num_iterations=3000,
                                     print_cost=True)

predictions_train = predict(train_x, train_y, learned_parameters)
predictions_test = predict(test_x, test_y, learned_parameters)
