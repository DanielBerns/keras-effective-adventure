from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


def build_LeNet(data_shape, 
                number_of_classes, 
                model_loss='categorical_crossentropy',
                model_optimizer='adadelta'):
    model = Sequential()
    model.add(Conv2D(32, 
                     kernel_size=(3, 3),
                     activation='relu',
                     input_shape=data_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), 
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, 
                    activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(number_of_classes, 
                    activation='softmax'))
    model.compile(loss=model_loss,
              optimizer=model_optimizer,
              metrics=['accuracy'])
    return model


def build_LeNet_5(data_shape, 
                  number_of_classes, 
                  model_loss='categorical_crossentropy',
                  model_optimizer='adam'):
    model = Sequential()
    model.add(Conv2D(6, 
                     kernel_size=(5, 5), 
                     activation='relu', 
                     input_shape=data_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), 
                           strides=(2, 2)))
    model.add(Conv2D(16, 
                     kernel_size=(5, 5),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, 
                    activation='relu'))
    model.add(Dense(84, 
                    activation='relu'))
    model.add(Dense(number_of_classes, 
                    activation='softmax'))
    model.compile(loss=model_loss, 
                  optimizer=model_optimizer, 
                  metrics = ['accuracy'])
    return model
