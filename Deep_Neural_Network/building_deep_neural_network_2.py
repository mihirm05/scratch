import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v3 import *


np.random.seed(1)


def L_layer_model(X, Y, layer_dims_val, learning_rate=0.0075, num_iterations=2500, print_cost=True):
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
        AL, cache_list = L_model_forward(X, parameters)

        cost = compute_cost(AL, Y)

        grads_list = L_model_backward(AL, Y, cache_list)

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

learned_parameters = L_layer_model(train_x, train_y, layer_dims, num_iterations=2500, print_cost=True)

pred_train = predict(train_x, train_y, learned_parameters)
pred_test = predict(test_x, test_y, learned_parameters)
print_mislabeled_images(classes, test_x, test_y, pred_test)

