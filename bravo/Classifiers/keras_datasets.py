from keras.datasets import mnist
from keras.datasets import cifar10
from keras.utils import to_categorical

import numpy as np


def get_keras_cifar10():
    print("# Dataset")
    print('##   load data')
    (train_X, train_y), (test_X, test_y) = cifar10.load_data()
    print('##   normalize data')
    train_X = ((train_X - 127).astype('float32') / 128)
    test_X = ((test_X - 127).astype('float32') / 128)
    print('##   perform one-hot encoding on the classes')
    train_y = to_categorical(train_y, num_classes=10)
    test_y = to_categorical(test_y, num_classes=10)
    data_shape = (32, 32, 3)
    labels = [str(v) for v in range(10)]
    return train_X, train_y, test_X, test_y, data_shape, labels





#----------------------------------------------------------------------
class KerasMNISTDataset:
    def __init__(self):
        self._expected_shape = (28, 28, 1)
        (train_X, train_y), (test_X, test_y) = mnist.load_data()
        self._images = (np.vstack([train_X, test_X]) - 127.0) / 128.0
        self._codes = np.hstack([train_y, test_y])
        self._set_of_labels = np.array([str(v) for v in range(10)])
        self._onehots = to_categorical(self._codes, num_classes=10)
        self._labels = self.set_of_labels[self._codes]        
        
    @property
    def expected_shape(self):
        return self._expected_shape
    
    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def codes(self):
        return self._codes
    
    @property
    def onehots(self):
        return self._onehots

    @property
    def set_of_labels(self):
        return self._set_of_labels

    def get_label(self, onehot):
        if len(onehot) == len(self.set_of_labels):
            return self.set_of_labels[onehot.argmax()]
        else:
            return "onehot.error"
        
    def get(self, branch=0.20, random_state=42):
        np.random.seed(random_state)
        train_X = list()
        train_y = list()
        validation_X = list()
        validation_y = list()
        test_X = list()
        test_y = list()
        train_branch = 1.0 - branch
        validation_branch = train_branch**2
        for image, onehot in zip(self.images, self.onehots):
            die = np.random.uniform()
            if die < validation_branch:
                train_X.append(image)
                train_y.append(onehot)
            elif die < train_branch:
                validation_X.append(image)
                validation_y.append(onehot)
            else:
                test_X.append(image)
                test_y.append(onehot)
        train_X = np.array(train_X)
        train_y = np.array(train_y)
        validation_X = np.array(validation_X)
        validation_y = np.array(validation_y)
        test_X = np.array(test_X)
        test_y = np.array(test_y)
        return train_X, train_y, validation_X, validation_y, test_X, test_y, self.expected_shape, self.set_of_labels


class ReshapedKerasMNISTDataset:
    def __init__(self):
        keras_mnist_dataset = KerasMNISTDataset()
        original_shape = keras_mnist_dataset.images.shape
        expected_shape = (original_shape[1]*original_shape[2],)
        images = keras_mnist_dataset.images.reshape((original_shape[0], expected_shape[0]))
        self._expected_shape = expected_shape
        self._images = images
        self._codes = keras_mnist_dataset.codes
        self._set_of_labels = keras_mnist_dataset.set_of_labels
        self._onehots = keras_mnist_dataset.onehots
        self._labels = keras_mnist_dataset.labels        
        
    @property
    def expected_shape(self):
        return self._expected_shape
    
    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def codes(self):
        return self._codes
    
    @property
    def onehots(self):
        return self._onehots

    @property
    def set_of_labels(self):
        return self._set_of_labels

    def get_label(self, onehot):
        if len(onehot) == len(self.set_of_labels):
            return self.set_of_labels[onehot.argmax()]
        else:
            return "onehot.error"
        
    def get(self, branch=0.20, random_state=42):
        np.random.seed(random_state)
        train_X = list()
        train_y = list()
        validation_X = list()
        validation_y = list()
        test_X = list()
        test_y = list()
        train_branch = 1.0 - branch
        validation_branch = train_branch**2
        for image, onehot in zip(self.images, self.onehots):
            die = np.random.uniform()
            if die < validation_branch:
                train_X.append(image)
                train_y.append(onehot)
            elif die < train_branch:
                validation_X.append(image)
                validation_y.append(onehot)
            else:
                test_X.append(image)
                test_y.append(onehot)
        train_X = np.array(train_X)
        train_y = np.array(train_y)
        validation_X = np.array(validation_X)
        validation_y = np.array(validation_y)
        test_X = np.array(test_X)
        test_y = np.array(test_y)
        return train_X, train_y, validation_X, validation_y, test_X, test_y, self.expected_shape, self.set_of_labels
