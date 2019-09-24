from Classifiers import Classifier
from Classifiers import KerasMNISTDataset
from Classifiers import build_LeNet_5


if __name__ == '__main__':
    dataset = KerasMNISTDataset()
    train_X, train_y, validation_X, validation_y, test_X, test_y, data_shape, set_of_labels = dataset.get()
    model = build_LeNet_5(data_shape, len(set_of_labels))
    classifier = Classifier()
    classifier.train_epochs = 1
    classifier.output = 'output/mnist/LeNet_5-alpha'
    classifier.train(model, train_X, train_y, test_X, test_y, set_of_labels, initial_alpha=0.001, factor=0.5)
