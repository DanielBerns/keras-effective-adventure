from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation

HIDDEN = 128
DROPOUT = 0.3
RESHAPED = 784

def build_MNIST_KerasNet(
    data_shape, number_of_classes,
    model_loss='categorical_crossentropy',
    model_optimizer='RMSprop'
    ):
    model = Sequential()
    model.add(Dense(HIDDEN, input_shape=data_shape))
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(HIDDEN))
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(number_of_classes))
    model.add(Activation('softmax'))
    model.compile(loss=model_loss,
              optimizer=model_optimizer,
              metrics=['accuracy'])
    model.summary()
    return model
