from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import BatchNormalization # https://arxiv.org/pdf/1502.03167v3.pdf
from keras.layers import Activation

from keras import regularizers

    
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


