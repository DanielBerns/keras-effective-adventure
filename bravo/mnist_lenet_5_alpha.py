import numpy as np

from Core import Classifier
from Datasets import get_keras_mnist
from Models import build_LeNet_5


if __name__ == '__main__':
    train_X, train_y, test_X, test_y, data_shape, labels = get_keras_mnist()
    model = build_LeNet_5(data_shape, len(labels))
    classifier = Classifier()
    classifier.train_epochs = 100
    classifier.output = 'output/mnist/lenet_5'
    classifier.train(model, train_X, train_y, test_X, test_y, labels)
