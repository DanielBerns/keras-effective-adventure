from Classifiers import Classifier
from Classifiers import get_keras_mnist
from Classifiers import build_LeNet


if __name__ == '__main__':
    train_X, train_y, test_X, test_y, data_shape, labels = get_keras_mnist()
    model = build_LeNet(data_shape, len(labels))
    classifier = Classifier()
    classifier.train_epochs = 1
    classifier.output = 'output/mnist/LeNet-alpha'
    classifier.train(model, train_X, train_y, test_X, test_y, labels)
