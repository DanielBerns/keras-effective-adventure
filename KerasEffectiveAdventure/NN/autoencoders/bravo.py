from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

from keras.datasets import mnist
import numpy as np



# this is the size of our encoded representations
encoding_size = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

#######################################################################
encoder_input = Input(shape=(784,))
encoder_layers = Dense(128, activation='relu')(encoder_input)
encoder_layers = Dense(64, activation='relu')(encoder_layers)
encoder_layers = Dense(encoding_size, activation='relu')(encoder_layers)
encoder = Model(encoder_input, encoder_layers, name='encoder')
print('-'*80)
encoder.summary()


decoder_input = Input(shape=(encoding_size,))
decoder_layers = Dense(64, activation='relu')(decoder_input)
decoder_layers = Dense(128, activation='relu')(decoder_layers)
decoder_layers = Dense(784, activation='sigmoid')(decoder_layers)
decoder = Model(decoder_input, decoder_layers, name='decoder')
print('-'*80)
decoder.summary()


autoencoder = Model(encoder_input, decoder(encoder(encoder_input)), name='autoencoder')
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
print('-'*80)
print(autoencoder.summary())

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(str(x_train.shape))
print(str(x_test.shape))

autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

