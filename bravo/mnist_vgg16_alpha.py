import numpy as np

from Core import Classifier
from Datasets import get_keras_mnist
from Models import build_VGG16


if __name__ == '__main__':
    train_X, train_y, test_X, test_y, data_shape, labels = get_keras_mnist()
    model = build_VGG16(data_shape=(256, 256, 3), num_classes=len(labels))
    classifier = Classifier()
    classifier.train_epochs = 1
    classifier.output = 'output/mnist/vgg16'
    classifier.train(model, train_X, train_y, test_X, test_y, labels)
