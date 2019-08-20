from Classifiers import Classifier
from Classifiers import get_keras_mnist
from Classifiers import build_AlphaNet


if __name__ == '__main__':
    train_X, train_y, test_X, test_y, data_shape, labels = get_keras_mnist()
    model = build_AlphaNet(data_shape, len(labels))
    classifier = Classifier()
    classifier.train_epochs = 100
    classifier.output = 'output/mnist/alphanet-alpha'
    classifier.train(model, train_X, train_y, test_X, test_y, labels)
