from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation

import numpy as np

from Core import Classifier
from Datasets import get_keras_mnist
from Models import build_LeNet


if __name__ == '__main__':
    train_X, train_y, test_X, test_y, data_shape, labels = get_keras_mnist()
    model = build_LeNet(data_shape, len(labels))
    classifier = Classifier()
    classifier.train_epochs = 100
    classifier.output = 'output/mnist/lenet'
    classifier.train(model, train_X, train_y, test_X, test_y, labels)
