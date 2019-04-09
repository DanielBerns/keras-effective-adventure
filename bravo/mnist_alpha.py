from keras.datasets.mnist import load_data
from keras.utils import to_categorical


def get_keras_mnist_dataset(parameters=None):
    # load data
    (train_X, train_y), (test_X, test_y) = load_data()
    # normalize data
    train_X = (train_X - 127).astype('float32') / 128
    test_X = (test_X - 127).astype('float32') / 128
    # perform one-hot encoding on the classes
    train_y = to_categorical(train_y, num_classes=10)
    test_y = to_categorical(test_y, num_classes=10)
    labels = [f"{v}" for v in range(10)]
    data_shape = (28, 28, 1)
    return train_X, train_y, test_X, test_y, labels, data_shape


def build_LeNet(datas_shape, classes):
    # initialize the model
    model = Sequential()

    # first set of CONV => RELU => POOL layers
    model.add(Conv2D(20, (5, 5), padding="same", input_shape=data_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL layers
    model.add(Conv2D(50, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    # return the constructed network architecture
    return model

if __name__ == '__main__':
    train_X, train_y, test_X, test_y, labels, data_shape = get_keras_mnist_dataset()
    model = build_LeNet(data_shape, length(labels))
    history = train(model, train_X, train_y, test_X, test_y, labels)
