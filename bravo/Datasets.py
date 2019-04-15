from keras.datasets import mnist
from keras.datasets import cifar10
from keras.utils import to_categorical

import numpy as np


def get_keras_mnist():
    print("# Dataset")
    print('##   load data')
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    print('##   normalize data')
    train_X = ((train_X - 127).astype('float32') / 128)[:, :, :, np.newaxis]
    test_X = ((test_X - 127).astype('float32') / 128)[:, :, :, np.newaxis]
    print('##   perform one-hot encoding on the classes')
    train_y = to_categorical(train_y, num_classes=10)
    test_y = to_categorical(test_y, num_classes=10)
    data_shape = (28, 28, 1)
    labels = [str(v) for v in range(10)]
    return train_X, train_y, test_X, test_y, data_shape, labels

 
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
