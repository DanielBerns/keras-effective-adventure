from Classifiers import Classifier
from Classifiers import get_keras_mnist
from Classifiers import build_BravoNet


if __name__ == '__main__':
    train_X, train_y, test_X, test_y, data_shape, labels = get_keras_mnist()
    model = build_BravoNet(
        data_shape, 
        len(labels), 
        filters=[32, 64, 128, 256], 
        enable_dropout=False, 
        enable_batch_normalization=True
        )
    classifier = Classifier()
    classifier.train_epochs = 20
    classifier.output = 'output/mnist/bravonet-alpha'
    classifier.train(model, train_X, train_y, test_X, test_y, labels)
