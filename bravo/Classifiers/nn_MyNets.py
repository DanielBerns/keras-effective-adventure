from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import BatchNormalization # https://arxiv.org/pdf/1502.03167v3.pdf

from keras import regularizers


def build_MyNetAlpha(data_shape, 
                number_of_classes, 
                model_loss='categorical_crossentropy',
                model_optimizer='adam'):
    model = Sequential()
    model.add(Conv2D(16, (3, 3), 
                     input_shape=data_shape, 
                     activation='selu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='selu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(number_of_classes * 16, activation='selu'))
    model.add(Dense(number_of_classes * 8, activation='selu'))
    model.add(Dense(number_of_classes, activation='softmax'))
    model.compile(loss=model_loss, 
                  optimizer=model_optimizer, 
                  metrics=['accuracy'])
    return model


def build_MyNetBravo(
    data_shape, 
    number_of_classes, 
    filters=None,
    activation_function='selu',
    enable_dropout=True, dropout=0.25,
    enable_batch_normalization=False,
    model_loss='categorical_crossentropy',
    model_optimizer='adam'):
    
    if filters==None:
        filters=[32, 64, 128, 256]
    channels_dimension = -1
    model = Sequential()
    model.add(Conv2D(filters[0], (3, 3), strides=(2, 2), 
                     input_shape=data_shape, 
                     activation=activation_function))
    if enable_dropout:
        model.add(Dropout(dropout))
    if enable_batch_normalization:
        model.add(BatchNormalization(axis=channels_dimension))        

    model.add(Conv2D(filters[1], (3, 3), strides=(2, 2),
                     activation=activation_function))
    if enable_dropout:
        model.add(Dropout(dropout))
    if enable_batch_normalization:
        model.add(BatchNormalization(axis=channels_dimension))        
        
    model.add(Conv2D(filters[2], (3, 3), strides=(2, 2),
                     activation=activation_function))
    if enable_dropout:
        model.add(Dropout(dropout))
    if enable_batch_normalization:
        model.add(BatchNormalization(axis=channels_dimension))        

    model.add(Conv2D(filters[3], (2, 2), padding='valid',
                     activation=activation_function))
    if enable_dropout:
        model.add(Dropout(dropout))
    if enable_batch_normalization:
        model.add(BatchNormalization(axis=channels_dimension))        

    model.add(Flatten())
    model.add(Dense(filters[3] * 4, activation='selu'))
    if enable_dropout:
        model.add(Dropout(dropout))
    if enable_batch_normalization:
        model.add(BatchNormalization(axis=channels_dimension))        

    model.add(Dense(filters[3], activation='selu'))
    if enable_dropout:
        model.add(Dropout(dropout))
    if enable_batch_normalization:
        model.add(BatchNormalization(axis=channels_dimension))        

    model.add(Dense(number_of_classes, activation='softmax'))
    model.compile(loss=model_loss, 
                  optimizer=model_optimizer, 
                  metrics=['accuracy'])
    return model

def build_MyNetCharlie(
    data_shape, 
    number_of_classes, 
    filters=None,
    activation_function='selu',
    enable_dropout=True, dropout=0.25,
    enable_batch_normalization=False,
    model_loss='categorical_crossentropy',
    model_optimizer='adam'):
    
    if filters==None:
        filters=[32, 64]
    channels_dimension = -1
    model = Sequential()
    model.add(Conv2D(filters[0], (3, 3), strides=(2, 2), 
                     input_shape=data_shape, 
                     activation=activation_function))
    if enable_dropout:
        model.add(Dropout(dropout))
    if enable_batch_normalization:
        model.add(BatchNormalization(axis=channels_dimension))        

    model.add(Conv2D(filters[1], (3, 3), strides=(2, 2),
                     activation=activation_function))
    if enable_dropout:
        model.add(Dropout(dropout))
    if enable_batch_normalization:
        model.add(BatchNormalization(axis=channels_dimension))        
        
    model.add(Flatten())
    model.add(Dense(filters[1] * 4, activation='selu'))
    if enable_dropout:
        model.add(Dropout(dropout))
    if enable_batch_normalization:
        model.add(BatchNormalization(axis=channels_dimension))        

    model.add(Dense(filters[1], activation='selu'))
    if enable_dropout:
        model.add(Dropout(dropout))
    if enable_batch_normalization:
        model.add(BatchNormalization(axis=channels_dimension))        

    model.add(Dense(number_of_classes, activation='softmax'))
    model.compile(loss=model_loss, 
                  optimizer=model_optimizer, 
                  metrics=['accuracy'])
    return model
