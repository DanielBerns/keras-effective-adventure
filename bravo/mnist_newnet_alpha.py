from Classifiers import Classifier
from Classifiers import get_keras_mnist
from Classifiers import build_NewNet


if __name__ == '__main__':
    train_X, train_y, test_X, test_y, data_shape, labels = get_keras_mnist()
    model = build_NewNet(data_shape, len(labels))
    classifier = Classifier()
    classifier.train_epochs = 5
    classifier.output = 'output/mnist/newnet'
    classifier.train(model, train_X, train_y, test_X, test_y, labels)
