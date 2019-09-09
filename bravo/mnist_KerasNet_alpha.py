from Classifiers import Classifier
from Classifiers import get_reshaped_keras_mnist
from Classifiers import build_MNIST_KerasNet

if __name__ == '__main__':
    train_X, train_y, test_X, test_y, data_shape, labels = get_reshaped_keras_mnist()
    model = build_MNIST_KerasNet(len(labels), data_shape)
    classifier = Classifier()
    classifier.train_epochs = 1
    classifier.output = 'output/mnist/KerasNet-alpha'
    classifier.train(model, train_X, train_y, test_X, test_y, labels)
