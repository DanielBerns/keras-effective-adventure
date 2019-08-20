+import numpy as np

from Classifiers import Classifier, get_keras_mnist, build_MiniVGGNet


if __name__ == '__main__':
    train_X, train_y, test_X, test_y, data_shape, labels = get_keras_mnist()
    model = build_MiniVGGNet((28, 28, 1), len(labels))
    classifier = Classifier()
    classifier.train_epochs = 20
    classifier.output = 'output/mnist/mini_vgg_net-alpha'
    classifier.train(model, train_X, train_y, test_X, test_y, labels)
