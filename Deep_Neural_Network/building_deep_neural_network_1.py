from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward
from dnn_app_utils_v3 import *

import numpy as np
import h5py
import matplotlib.pyplot as plt

np.random.seed(1)


def initialize_parameters_deep(layer_dims):
    """
    :param layer_dims: dimensions of the intermediate layers
    :return: initialized weights and biases
    """
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for layer in range(1, L):
        parameters["W" + str(layer)] = np.random.randn(layer_dims[layer], layer_dims[layer - 1]) * 0.01
        parameters["b" + str(layer)] = np.zeros((layer_dims[layer], 1))

    return parameters


def linear_forward(A, W, b):
    """
    :param A: activation values
    :param W: weight matrix
    :param b: bias matrix
    :return: Z: input for activation function
    cache: tuple containing (A, W, b), used during the backward pass
    """
    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A, W, b, activation):
    """
    :param A: activation from previous layers
    :param W: weight values
    :param b: bias values
    :param activation: activation function
    :return:
    A: output of activation function
    cache: tuple containing linear and activation cache
    """
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A, W, b)
        A, activation_cache = sigmoid(Z)

    else:
        Z, linear_cache = linear_forward(A, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    """
    :param parameters: weight and biases
    :param X: data
    :return:
    AL: post activation value
    cache: cache containing linear and activation cache
    """
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], activation="relu")
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], activation="sigmoid")
    caches.append(cache)

    return AL, caches


def compute_cost(AL, Y):
    """
    :param AL: probability of final vector
    :param Y: true label
    :return:
    cross entropy cost
    """
    m = Y.shape[1]

    cost = -(np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))) / m
    cost = np.squeeze(cost)
    return cost


def linear_backward(dZ, cache):
    """
    :param dZ: gradient of cost wrt Z
    :param cache: tuple containing (A_prev, W, b) coming from forward pass
    :return:
    dA_prev: gradient of cost wrt previous layer activation
    dW: gradient of cost wrt W
    db: gradient of cost wrt b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (np.dot(dZ, cache[0].T)) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(cache[1].T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, cache):
    """
    :param AL: output of forward propagation
    :param Y: true labels
    :param cache: list of caches (l-1) output for relu and cache l output for sigmoid
    :return:
    grads: gradients for dA, dW and db
    """
    grads = {}
    L = len(cache)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = cache[-1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward(sigmoid_backward(dAL, current_cache[1]), current_cache[0])

    for layers in reversed(range(L - 1)):
        current_cache = cache[layers]
        dA_prev_temp, dW_temp, db_temp = linear_backward(relu_backward(grads["dA" + str(layers + 1)], current_cache[1]), current_cache[0])

        grads["dA" + str(layers)] = dA_prev_temp
        grads["dW" + str(layers+1)] = dW_temp
        grads["db" + str(layers+1)] = db_temp
        np.set_printoptions(suppress=True)
    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    :param parameters: weights and biases
    :param grads: gradients of cost wrt parameters
    :param learning_rate: learning rate
    :return:
    updated parameters
    """
    L = len(parameters) // 2

    for layers in range(L):
        parameters["W"+str(layers+1)] = parameters["W"+str(layers+1)]-learning_rate*grads["dW"+str(layers+1)]
        parameters["b"+str(layers+1)] = parameters["b"+str(layers+1)]-learning_rate*grads["db"+str(layers+1)]

    return parameters


