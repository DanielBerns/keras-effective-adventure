from Classifiers import Classifier
from Classifiers import get_keras_mnist
from Classifiers import build_MyNetCharlie


if __name__ == '__main__':
    train_X, train_y, test_X, test_y, data_shape, labels = get_keras_mnist()
    model = build_MyNetCharlie(
        data_shape, 
        len(labels), 
        filters=[64, 128], 
        enable_dropout=False, 
        enable_batch_normalization=True
        )
    classifier = Classifier()
    classifier.train_epochs = 1
    classifier.output = 'output/mnist/charlienet-alpha'
    classifier.train(model, train_X, train_y, test_X, test_y, labels)
