import glob
import matplotlib.pyplot as plt 
import numpy as np 
import imageio as im 
import keras 
import matplotlib.image as mpimg 

from keras import models
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from keras.preprocessing.image import load_img, img_to_array 
from keras.preprocessing.image import ImageDataGenerator 
from keras.callbacks import ModelCheckpoint 


def dataLoader(path):
    """ load and display the images stored at the given path"""

    images = [] 
    
    for img_path in glob.glob(path):
        images.append(mpimg.imread(img_path))
    plt.figure(figsize=(20,10))
    columns = 5 

    for i, image in enumerate(images): 
        plt.subplot(len(images) / columns+1, columns, i+1)
        plt.imshow(image)
    plt.show()


def buildModel(classifier):
    """returns the model"""
    
    # convolution 1
    classifier.add(Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 3), activation='relu'))
    classifier.add(Conv2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.5))

    # convolution 2
    classifier.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.5))

    # convolution 3 
    classifier.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.5))

    # flatten 
    classifier.add(Flatten())

    # fully connected 
    classifier.add(Dense(units=512, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units=3, activation='softmax'))


def compileModel(model, optimizer, loss, metrics):
    """compiles the input model"""
    model.compile(optimizer, loss, metrics)
    # print('Model compiled successfully')


def splitData():
    """splits the data into test and train sets"""
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255) 
    
    training_set = train_datagen.flow_from_directory('training_set', target_size=(28, 28), 
                                                     batch_size=16, class_mode='categorical')
    test_set = test_datagen.flow_from_directory('test_set', target_size=(28,28), 
                                                batch_size=16, class_mode='categorical') 

    return training_set, test_set 


def plotGraphs(history):
    """plot visualization for accuracy and loss """

    acc = history.history['acc']
    val_acc = history.history['val_acc']

    loss = history.history['loss']
    val_loss = history.history['val_loss'] 

    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')

    plt.title('Training and validation accuracy')
    plt.legend() 
    plt.figure() 

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show() 
    

def predict(img_path, classifier):
    """predict the class of input"""
    img = load_img(img_path, target_size=(28, 28))
    img_tensor = img_to_array(img) 
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    plt.imshow(img_tensor[0])
    plt.show()
    print(img_tensor.shape)

    x = img_to_array(img)
    x = np.expand_dims(x, axis=0) 
    images = np.vstack([x])
    classes = classifier.predict_classes(images, batch_size=10)

    if classes == 0:
        print('predicted class is circle' ) 

    elif classes == 1:
        print('predicted class is square')

    else:
        print('predicted class is triangle')

    return img_tensor


def visualizingFeatures(classifier, img_tensor):
    """visualizes the features extracted by the neural network layers""" 

    layer_outputs = [layer.output for layer in classifier.layers[:12]]
    print() 
    print(layer_outputs) 
    print() 

    # extracts the outputs of the top 12 models 
    activation_model = models.Model(inputs=classifier.input, outputs=layer_outputs)

    activations = activation_model.predict(img_tensor) 
    first_layer_activation = activations[0] 
    plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')

    layer_names = [] 
    for layer in classifier.layers[:12]:
        layer_names.append(layer.name)

    images_per_row = 16 

    for layer_name, layer_activation in zip(layer_names, activations):  # displaying feature maps
        n_features = layer_activation.shape[-1]    # number of features in the feature map 
        size = layer_activation.shape[1]    # feature map has shape (1, size, size, n_features) 
        n_cols = n_features // images_per_row    # activation channels in this matrix 
        display_grid = np.zeros((size*n_cols, images_per_row*size))

        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col*images_per_row + row]
                # channel_image -= channel_image.mean() # make features visually clean 
                # print(channel_image)
                # channel_image /= channel_image.std()
                channel_image *= 64 
                channel_image += 128 
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col*size : (col+1)*size, row*size : (row+1)*size] = channel_image
        
        scale = 1./size 
        plt.figure(figsize=(scale*display_grid.shape[1], scale*display_grid.shape[0]))

        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis') 
    plt.show()


def main():
    dataLoader('training_set/circles/*.png')
    dataLoader('training_set/squares/*.png')
    dataLoader('training_set/triangles/*.png')
    
    classifier = Sequential() 
    buildModel(classifier)
    classifier.summary() 

    compileModel(classifier, 'rmsprop', 'categorical_crossentropy', ['accuracy'])

    trainData, testData = splitData()

    checkpointer = ModelCheckpoint(filepath='best_weights.hdf5', monitor='val_acc', 
                                   save_best_only=True)

    history = classifier.fit_generator(trainData, steps_per_epoch=100,
                                       epochs=20, callbacks=[checkpointer],
                                       validation_data=testData, validation_steps=50)

    classifier.load_weights('best_weights.hdf5') 
    classifier.save('shapes_cnn.h5')

    plotGraphs(history)
    img_tensor = predict('test_set/triangles/drawing(72).png', classifier)
    visualizingFeatures(classifier, img_tensor)
    

if __name__ == '__main__':
    main()
    
