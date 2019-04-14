from Core import Classifier

from keras.datasets.cifar10 import load_data
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation

import numpy as np

def get_dataset(parameters=None):
    print("# Dataset")
    print('##   load data')
    (train_X, train_y), (test_X, test_y) = load_data()
    print('##   normalize data')
    train_X = ((train_X - 127).astype('float32') / 128)
    test_X = ((test_X - 127).astype('float32') / 128)
    print('##   perform one-hot encoding on the classes')
    train_y = to_categorical(train_y, num_classes=10)
    test_y = to_categorical(test_y, num_classes=10)
    labels = [str(v) for v in range(10)]
    data_shape = (32, 32, 3)
    return train_X, train_y, test_X, test_y, data_shape, labels


def build_model(data_shape, num_classes):
    print('# Model')
    print("##   Build")
    model = Sequential()
    model.add(Conv2D(16, (3, 3), strides=(2, 2), padding='same',
                     input_shape=data_shape))
    model.add(Activation('selu'))
    model.add(Conv2D(16, (3, 3), strides=(2, 2)))
    model.add(Activation('selu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(Activation('selu'))
    model.add(Conv2D(64, (3, 3), strides=(2, 2)))
    model.add(Activation('selu'))
    
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('selu'))
    
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    print("##   Compile")
    model.compile(optimizer='adadelta', loss="categorical_crossentropy", metrics = ['accuracy'])

    model.summary()
    
    return model


if __name__ == '__main__':
    train_X, train_y, test_X, test_y, data_shape, labels = get_dataset()
    model = build_model(data_shape, len(labels))
    classifier = Classifier()
    classifier.train_epochs = 100
    classifier.output = 'delta'
    classifier.train(model, train_X, train_y, test_X, test_y, labels)
