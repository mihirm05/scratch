import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random


def buildModel(learningRate):
    """
    Create a simple Linear Regression model
    :param learningRate: learning rate
    :return: Linear regression model
    """
    # a sequential model as it contains one or more layers
    model = tf.keras.models.Sequential()

    # a linear regression model has a single node in a single layer
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

    # compiles a model into code so that tf can execute it. configuring the training to minimize rmse
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learningRate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def trainModel(model, feature, label, epochs, batch_size):
    """
    Train the model by feeding data to it
    :param model: defined in the buildModel function
    :param feature: features of the input data
    :param label: label of the input data
    :param epochs: number of epochs to be trained
    :param batch_size: batch size to be used
    :return:  trained weights, trained biases, epochs and RMSE
    """
    # model trains for the specified number of epochs and learns how the feature values map to the label values
    history = model.fit(x=feature,
                        y=label,
                        batch_size=batch_size,
                        epochs=epochs)

    # extract trained weights and bias
    trainedWeight = model.get_weights()[0]
    trainedBias = model.get_weights()[1]

    # store the number of epochs as well
    epochs = history.epoch

    # gather a snapshot of each epoch
    hist = pd.DataFrame(history.history)

    # gather root mean square error at each epoch
    rmse = hist["root_mean_squared_error"]

    return trainedWeight, trainedBias, epochs, rmse


def plotModel(trained_weight, trained_bias, feature, label):
    """
    :param trained_weight: learned weights
    :param trained_bias: learned biases
    :param feature: input data features
    :param label: input data labels
    :return: plot the trained model against the training feature and label
    """

    plt.xlabel('feature')
    plt.ylabel('label')
    plt.scatter(feature, label)

    # create a red line representing model that begins at (x0, y0) and ends at (x1, y1)
    x0 = 0
    y0 = trained_bias
    x1 = feature[-1]
    y1 = trained_bias + trained_weight * x1
    plt.plot([x0, x1], [y0, y1], c='r')

    plt.savefig('fit.pdf')


def plotLossCurve(epochs, rmse):
    """
    plot the loss curve which shows the epochs vs loss relation
    :param epochs: number of epochs
    :param rmse: rmse for corresponding epoch
    :return: epoch vs rmse plot
    """
    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('rmse')

    plt.plot(epochs, rmse, label='loss')
    plt.legend()

    plt.ylim([rmse.min() * 0.97, rmse.max()])
    plt.savefig('loss.pdf')


def main():

    feature = list(np.linspace(1, 12, 12))
    print(feature)
    # ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])

    label = list(random.sample(range(0, 50), 12))
    label.sort()
    # ([5.0, 8.8, 9.6, 14.2, 18.8, 19.5, 21.4, 26.8, 28.9, 32.0, 33.8, 38.2])

    learning_rate = 0.14
    epochs = 70
    batch_size = 12

    model = buildModel(learning_rate)
    trained_weights, trained_bias, epochs, rmse = trainModel(model, feature, label, epochs, batch_size)

    plotModel(trained_weights, trained_bias, feature, label)
    plotLossCurve(epochs, rmse)


if __name__ == "__main__":
    main()
