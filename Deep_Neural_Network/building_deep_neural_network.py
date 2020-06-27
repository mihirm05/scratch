from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
from dnn_app_utils import *

import numpy as np
import h5py
import matplotlib.pyplot as plt

# np.random.seed(1)


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

    elif activation == "relu":
        Z, linear_cache = linear_forward(A, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache


def l_model_forward(X, parameters):
    """
    :param parameters_val: weight and biases
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


def l_model_backward(AL, Y, cache):
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


def l_layer_model(X, Y, layer_dims_val, learning_rate=0.0075, num_iterations=2500, print_cost=True):
    """
    consolidates all the functions in a single unit
    :param X: input images
    :param Y: true labels
    :param layer_dims_val: architecture of the model
    :param learning_rate: learning_rate
    :param num_iterations: number of iterations
    :param print_cost: print cost of iterations
    :return:
    updated_parameters: learned parameters after training
    """
    np.random.seed(1)
    costs = []

    parameters = initialize_parameters_deep(layer_dims)

    for i in range(num_iterations+1):
        AL, cache_list = l_model_forward(X, parameters)

        cost = compute_cost(AL, Y)

        grads_list = l_model_backward(AL, Y, cache_list)

        parameters = update_parameters(parameters, grads_list, learning_rate)

        if print_cost and i % 100 == 0:
            print('COST AFTER ITERATION %i:%f' % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title('Learning rate = ' + str(learning_rate))
    plt.grid()
    plt.show()

    return parameters


train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
train_x = train_x_flatten / 255
test_x = test_x_flatten / 255

layer_dims = [12288, 20, 7, 5, 1]

parameters = l_layer_model(train_x, train_y, layer_dims, num_iterations=2500, print_cost=True)

pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)
print_mislabeled_images(classes, test_x, test_y, pred_test)

