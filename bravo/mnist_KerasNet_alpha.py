from Classifiers import Classifier
from Classifiers import ReshapedKerasMNISTDataset
from Classifiers import build_MNIST_KerasNet

# import pdb

if __name__ == '__main__':
    dataset = ReshapedKerasMNISTDataset()
    train_X, train_y, validation_X, validation_y, test_X, test_y, data_shape, set_of_labels = dataset.get()
    model = build_MNIST_KerasNet(data_shape, len(set_of_labels))
    classifier = Classifier()
    classifier.train_epochs = 500
    classifier.output = 'output/mnist/KerasNet-alpha'
    # pdb.set_trace()
    classifier.build(model, 
                     train_X, train_y, test_X, test_y, set_of_labels,
                     validation_X=validation_X, validation_y=validation_y)
