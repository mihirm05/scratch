from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
# index = 15
# plt.imshow(train_set_x_orig[index])
# plt.interactive(True)
# plt.show()
# print("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(
# train_set_y[:, index])].decode("utf-8") + "' picture.")

train_examples = train_set_x_orig.shape[0]
test_example = test_set_x_orig.shape[0]
number_pixel = train_set_x_orig.shape[1]

train_set_x_flatten = train_set_x_orig.reshape(number_pixel * number_pixel * train_set_x_orig.shape[3],
                                               train_set_x_orig.shape[0])
test_set_x_flatten = test_set_x_orig.reshape(number_pixel * number_pixel * test_set_x_orig.shape[3],
                                             test_set_x_orig.shape[0])

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255


def sigmoid(z):
    """
    Computes sigmoid of z
    :param z: input array or scalar
    :return: sigmoid(z)
    """
    s = 1 / (1 + np.exp(-z))
    return s


def initialize_with_zeros(dim):
    """
    Initialize the weight and bias of dimensions (dim,1)
    :param dim: size of w and b we want
    :return: initialized w and b
    """
    w = np.zeros((dim, 1))
    b = 0
    return w, b


def propagate(w, b, X, Y):
    """
    Implement forward pass and backward pass
    :param w: weight values
    :param b: bias values
    :param X: input features
    :param Y: true label
    :return:
    cost: negative log likelihood cost of LR
    dw: gradient of loss wrt w
    db: gradient of loss wrt b
    """
    m = X.shape[1]
    a = sigmoid(np.dot(w.T, X) + b)
    cost_val = -np.sum(Y * np.log(a) + (1 - Y) * np.log(1 - a)) / m
    dw = (np.dot(X, (a - Y).T)) / m
    db = np.sum(a - Y) / m

    grads = {"dw": dw,
             "db": db}
    return grads, cost_val


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    update weight and bias values
    :param print_cost: print the value of cost
    :param w: weight values
    :param b: bias values
    :param X: input values
    :param Y: output values
    :param learning_rate: update factor
    :param num_iterations: number of iterations
    :return:
    params: updated parameters
    grads: gradients
    costs: cost values
    """

    cost_val = []
    for i in range(num_iterations):
        grads, costing = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            cost_val.append(costing)

        if print_cost and i % 100 == 0:
            print('cost after iteration %i %f' % (i, costing))

    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}

    return params, grads, cost_val


def predict(w, b, X):
    """
    predict the input test image
    :param w: weight values
    :param b: bias values
    :param X: test input
    :return: predicted label
    """
    m = X.shape[1]
    y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    a = sigmoid(np.dot(w.T, X) + b)
    for i in range(a.shape[1]):
        y_prediction[0, i] = 1 if a[0, i] > 0.5 else 0

    return y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    builds the regression model
    :param print_cost: print cost or not
    :param X_train: input training set
    :param Y_train: input labels
    :param X_test: test set x values
    :param Y_test: test labels
    :param num_iterations: hyper parameter
    :param learning_rate: hyper parameter
    :return: dictionary containing model details
    """

    w, b = initialize_with_zeros(X_train.shape[0])
    parameters, grad, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w = parameters["w"]
    b = parameters["b"]

    y_prediction_test = predict(w, b, X_test)
    y_prediction_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - Y_test)) * 100))

    diction = {"cost": costs,
               "Y_prediction_test": y_prediction_test,
               "Y_prediction_train": y_prediction_train,
               "w": w,
               "b": b,
               "learning_rate": learning_rate,
               "num_iterations": num_iterations}

    return diction


d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=5000, learning_rate=0.0001, print_cost=True)
# print(test_set_y)
index = 0
# plt.imshow(test_set_x[:, index].reshape((number_pixel, number_pixel, 3))) print("y = " + str(test_set_y[0,
# index]) + ", you predicted that it is a \"" + str(classes[int(d["Y_prediction_test"][0, index])]) + "\" picture.")

cost = np.squeeze(d['cost'])
plt.plot(cost)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.grid()
plt.show()
