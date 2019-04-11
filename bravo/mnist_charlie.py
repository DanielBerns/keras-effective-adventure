from Core import Classifier

from keras.datasets.mnist import load_data
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Activation

import numpy as np

def get_keras_mnist_dataset(parameters=None):
    print("# Dataset")
    print('##   load data')
    (train_X, train_y), (test_X, test_y) = load_data()
    print('##   normalize data')
    train_X = ((train_X - 127).astype('float32') / 128)[:, :, :, np.newaxis]
    test_X = ((test_X - 127).astype('float32') / 128)[:, :, :, np.newaxis]
    print('##   perform one-hot encoding on the classes')
    train_y = to_categorical(train_y, num_classes=10)
    test_y = to_categorical(test_y, num_classes=10)
    labels = [v for v in range(10)]
    data_shape = (28, 28, 1)
    return train_X, train_y, test_X, test_y, data_shape, labels


def build_MiniVGGNet(data_shape, num_classes):
    print('# MiniVGGNet')
    print("##   Build")
    # initialize the model along with the input shape to be
    # "channels last" and the channels dimension itself
    model = Sequential()
    channels_dimension = -1

    # first CONV => RELU => CONV => RELU => POOL layer set
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=data_shape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channels_dimension))
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channels_dimension))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # second CONV => RELU => CONV => RELU => POOL layer set
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channels_dimension))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channels_dimension))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    print("##   Compile")
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics = ['accuracy'])

    return model


if __name__ == '__main__':
    train_X, train_y, test_X, test_y, data_shape, labels = get_keras_mnist_dataset()
    model = build_MiniVGGNet(data_shape, len(labels))
    classifier = Classifier()
    classifier.train_epochs = 100
    classifier.output = 'charlie'
    classifier.train(model, train_X, train_y, test_X, test_y, labels)