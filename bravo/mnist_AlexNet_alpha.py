import numpy as np

from Classifiers import Classifier, get_keras_mnist, build_AlexNet


if __name__ == '__main__':
    train_X, train_y, test_X, test_y, data_shape, labels = get_keras_mnist()
    model = build_AlexNet((28, 28, 1), len(labels))
    classifier = Classifier()
    classifier.train_epochs = 1
    classifier.output = 'output/mnist/AlexNet-alpha'
    classifier.train(model, train_X, train_y, test_X, test_y, labels)
