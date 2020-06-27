import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

X, Y = load_planar_dataset()


def data_visualization(X_val, Y_val):
    """
    plot the data points
    :param Y_val: class of the plotted points
    :param X_val: coordinates of the points
    :return:
    """
    plt.scatter(X_val[0, :], X_val[1, :], c=Y_val.ravel(), s=40, cmap=plt.cm.Spectral)
    plt.grid()
    plt.show()


def logistic_regression(X_val, Y_val):
    """
    performs logistic regression on the input values
    :param X_val: coordinates of the points
    :param Y_val: class of the plotted points
    :return:
    """
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X_val.T, Y_val.T)
    plot_decision_boundary(lambda x: clf.predict(x), X_val, Y_val)
    plt.title('Regression')
    LR_predictions = clf.predict(X_val.T)
    print('Accuracy of logistic regression: %d ' % float(
        (np.dot(Y_val, LR_predictions) + np.dot(1 - Y_val, 1 - LR_predictions)) / float(Y_val.size) * 100) +
          '% ' + "(percentage of correctly labelled datapoints)")


# logistic_regression(X, Y)
# data_visualization(X, Y)


def layer_size(X_val, Y_val):
    """
    sets the model structure
    :param X_val: coordinates of the point
    :param Y_val: class of the points
    :return: returns a tuple containing the number of units in each layer
    """
    n_x = X_val.shape[0]
    n_h = 4
    n_y = Y_val.shape[0]
    return n_x, n_h, n_y


def initialize_parameters(n_x_val, n_h_val, n_y_val):
    """
    initialize values
    :param n_x_val:  units in input layer
    :param n_h_val:  units in hidden layer
    :param n_y_val:  units in output layer
    :return: initialized parameters
    """
    np.random.seed(2)

    W1 = np.random.randn(n_h_val, n_x_val) * 0.01
    b1 = np.zeros((n_h_val, 1))
    W2 = np.random.randn(n_y_val, n_h_val) * 0.01
    b2 = np.zeros((n_y_val, 1))

    param = {"W1": W1,
             "b1": b1,
             "W2": W2,
             "b2": b2}
    return param


def forward_propagation(X_val, parameters):
    """
    calculate the forward pass
    :param X_val: coordinates of points
    :param parameters: weight and bias values
    :return: predicted values and cache
    """

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X_val) + b1
    a1 = np.tanh(Z1)
    Z2 = np.dot(W2, a1) + b2
    a2 = sigmoid(Z2)

    cache_list = {"Z1": Z1,
                  "a1": a1,
                  "Z2": Z2,
                  "a2": a2
                  }
    return a2, cache_list


def compute_cost(a2_val, Y_val, parameters):
    """
    computes the error function
    :param parameters: weights and bias
    :param a2_val: predicted values during the forward pass
    :param Y_val: class label in the train set
    :return: cost function value
    """
    m = Y_val.shape[1]
    log_probability = np.multiply(np.log(a2_val), Y_val) + np.multiply((1 - Y_val), np.log(1 - a2_val))
    cost = -np.sum(log_probability) / m
    cost = float(np.squeeze(cost))  # turns [[17]] to 17
    return cost


def backward_propagation(parameters, cache, X_val, Y_val):
    """
    process for the backward pass
    :param parameters: weight and bias values
    :param cache: Z and a values calculated in the forward pass
    :param X_val: coordinates of the points
    :param Y_val: class labels of the points
    :return: gradients for parameters
    """

    m = X_val.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    a1 = cache["a1"]
    a2 = cache["a2"]

    dZ2 = a2 - Y_val
    dW2 = (np.dot(dZ2, a1.T)) / m
    db2 = (np.sum(dZ2, axis=1, keepdims=True)) / m
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(a1, 2))
    dW1 = (np.dot(dZ1, X_val.T)) / m
    db1 = (np.sum(dZ1, axis=1, keepdims=True)) / m

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2,
             }
    return grads


def update_parameters(parameters, grads, learning_rate=1.2):
    """
    updates the weights and biases on the basis of gradients and learning rate
    :param parameters: weights and bias values
    :param grads: gradients of parameters
    :param learning_rate: hyper parameters
    :return: updated parameters
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2
                  }
    return parameters


def model(X_val, Y_val, num_iterations=1000, print_cost=True):
    """
    integrates the defined functions
    :param X_val: coordinates of points
    :param Y_val: class labels
    :param num_iterations: number of times to be repeated
    :param print_cost: cost to be printed or not
    :return: parameters learned by the model
    """
    np.random.seed(3)
    n_x, n_h, n_y = layer_size(X_val, Y_val)
    parameters = initialize_parameters(n_x, n_h, n_y)
    for i in range(0, num_iterations):
        a2, cache = forward_propagation(X_val, parameters)
        cost = compute_cost(a2, Y_val, parameters)
        grads = backward_propagation(parameters, cache, X_val, Y_val)
        parameters = update_parameters(parameters, grads, learning_rate=1.2)

        if print_cost and i % 100 == 0:
            print("cost after iteration %i: %f" % (i, cost))

    return parameters


def predict(updated_params, X_val):
    """
    predicts the class label based on the parameters and input
    :param updated_params: obtained from model()
    :param X_val: test x input values
    :return: predicted values based on the learned parameters
    """
    a2, cache = forward_propagation(X_val, updated_params)
    prediction = np.round(a2)
    return prediction


# Build a model with a n_h-dimensional hidden layer
updated_parameters = model(X, Y, num_iterations=10000, print_cost=True)

# Plot the decision boundary
plot_decision_boundary(lambda x: predict(updated_parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))

predictions = predict(updated_parameters, X)
print('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')