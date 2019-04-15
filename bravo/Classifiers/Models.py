from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import BatchNormalization # https://arxiv.org/pdf/1502.03167v3.pdf
from keras.layers import Activation

def build_LeNet(data_shape, 
                num_classes, 
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
    model.add(Dense(num_classes, 
                    activation='softmax'))
    model.compile(loss=model_loss,
              optimizer=model_optimizer,
              metrics=['accuracy'])
    return model


def build_LeNet_5(data_shape, 
                  num_classes, 
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
    model.add(Dense(num_classes, 
                    activation='softmax'))
    model.compile(loss=model_loss, 
                  optimizer=model_optimizer, 
                  metrics = ['accuracy'])
    return model


def build_MyNet(data_shape, 
                num_classes, 
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
    model.add(Dense(num_classes * 16, activation='selu'))
    model.add(Dense(num_classes * 8, activation='selu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=model_loss, 
                  optimizer=model_optimizer, 
                  metrics=['accuracy'])
    return model
    
def build_NewNet(data_shape, 
                 num_classes, 
                 filters_0=32, 
                 filters_1=64,
                 filters_2=128,
                 activation_function='selu',
                 dropout=0.25,
                 model_loss='categorical_crossentropy',
                 model_optimizer='adam'):
    model = Sequential()
    model.add(Conv2D(filters_0, (3, 3), strides=(2, 2), 
                     input_shape=data_shape, 
                     activation=activation_function))
    model.add(Dropout(dropout))
    model.add(Conv2D(filters_1, (3, 3), strides=(2, 2),
                     activation=activation_function))
    model.add(Dropout(dropout))
    model.add(Conv2D(filters_2, (3, 3), strides=(2, 2),
                     activation=activation_function))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(num_classes * 16, activation='selu'))
    model.add(Dense(num_classes * 8, activation='selu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=model_loss, 
                  optimizer=model_optimizer, 
                  metrics=['accuracy'])
    return model
    
def build_AlexNet(data_shape=(224, 224, 3),
                  num_classes=100,
                  model_loss='categorical_crossentropy',
                  model_optimizer='adam'):
    """
    https://www.mydatahack.com/building-alexnet-with-keras/
    """
    model = Sequential()
    
    # 1st Convolutional Layer
    model.add(Conv2D(96, 
                     kernel_size=(11, 11),
                     strides=(4, 4), 
                     padding='valid', 
                     input_shape=data_shape))
    model.add(Activation('relu'))
    # Pooling 
    model.add(MaxPooling2D(pool_size=(2, 2), 
                           strides=(2, 2), 
                           padding='valid'))
    # Batch Normalisation before passing it to the next layer
    model.add(BatchNormalization())
    
    # 2nd Convolutional Layer
    model.add(Conv2D(256, 
                     kernel_size=(11, 11), 
                     padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), 
                           strides=(2, 2), 
                           padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    # 3rd Convolutional Layer
    model.add(Conv2D(384, 
                     kernel_size=(3, 3), 
                     padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    # 4th Convolutional Layer
    model.add(Conv2D(384, 
                     kernel_size=(3, 3), 
                     padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    # 5th Convolutional Layer
    model.add(Conv2D(256, 
                     kernel_size=(3, 3), 
                     padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), 
                           strides=(2, 2), 
                           padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Passing it to a dense layer    
    model.add(Flatten())
    # 1st Dense Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    # 2nd Dense Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    # 3rd Dense Layer
    model.add(Dense(1000))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    # Output Layer
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    # Compile 
    model.compile(loss=model_loss, 
                  optimizer=model_optimizer, 
                  metrics=['accuracy'])

    return model


def build_VGG_16(data_shape=(224, 224, 3),
                 num_classes=1000,
                 model_loss='categorical_crossentropy',
                 model_optimizer='adam'):
    model = Sequential()
    
    model.add(ZeroPadding2D((1, 1), input_shape=data_shape))
    model.add(Conv2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', name='dense_1'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='dense_2'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, name='dense_3'))
    model.add(Activation('softmax', name='softmax'))

    model.compile(loss=model_loss,
                  optimizer=model_optimizer,
                  metrics=['accuracy'])

    return model
