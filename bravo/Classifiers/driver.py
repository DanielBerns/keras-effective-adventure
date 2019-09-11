# import the necessary packages
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import BaseLogger, LearningRateScheduler, ModelCheckpoint


def print_summary(target, model):
    # https://stackoverflow.com/questions/41665799/keras-model-summary-object-to-string
    strings = []
    model.summary(print_fn=lambda x: strings.append(x))
    model_summary = "\n".join(strings)
    target.write(model_summary)


def print_classification_report(y_true, y_predicted, labels):
    print("Print Classification Report")
    print(classification_report(y_true, y_predicted, target_names=labels))
    value = accuracy_score(y_true, y_predicted)
    print("Accuracy : {0:s}".format(str(value)))


def plot_confusion_matrix(y_true, y_predicted):
    print("Plot Confusion Matrix")
    mtx = confusion_matrix(y_true, y_predicted)
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(mtx, annot=True, fmt='d', linewidths=.5,  cbar=False, ax=ax)
    plt.ylabel('true label')
    plt.xlabel('predicted label')


def plot_history_loss_and_accuracy(history):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].plot(history.history['loss'])
    axs[0].plot(history.history['val_loss'])
    axs[0].set_title('model loss')
    axs[0].set_ylabel('loss')
    axs[0].set_xlabel('epoch')
    axs[0].legend(['train', 'validation'], loc='upper left')
    axs[1].plot(history.history['acc'])
    axs[1].plot(history.history['val_acc'])
    axs[1].set_title('model accuracy')
    axs[1].set_ylabel('accuracy')
    axs[1].set_xlabel('epoch')
    axs[1].legend(['train', 'validation'], loc='upper left')
    axs[2].plot(history.history['lr'])
    axs[2].set_title('learning rate')
    axs[2].set_ylabel('lr')
    axs[2].set_xlabel('epoch')


def print_history_keys(history):
    print('History keys')
    print(str(history.history.keys()))


def build_step_decay(initial_alpha, factor, drop_every=4):
    def step_decay(epoch):
        # compute learning rate for the current epoch
        step = np.floor((1 + epoch) / drop_every)
        alpha = initial_alpha * (factor ** step)
        return float(alpha) # learning rate
    return step_decay
    
    
class Classifier:
    def __init__(self):
        self._train_batch_size = 32
        self._train_epochs = 2
        self._verbose = 1
        self._output = 'output'
        self._predict_batch_size = 32

    @property
    def train_batch_size(self):
        return self._train_batch_size

    @property
    def train_epochs(self):
        return self._train_epochs

    @train_epochs.setter
    def train_epochs(self, value):
        self._train_epochs = value
    
    @property
    def verbose(self):
        return self._verbose
    
    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, value):
        self._output = value
    
    @property
    def predict_batch_size(self):
        return self._predict_batch_size
    
    def report(self, 
               model, 
               train_X, 
               train_y,
               test_X, 
               test_y, 
               labels,
               validation_split=None,
               validation_X=None, 
               validation_y=None, 
               initial_alpha=0.001, 
               factor=0.2):
        print("# Classifier")
        # construct the callback to save only the *best* model to disk
        # based on the validation loss
        output_path = Path(self.output)
        output_path.mkdir(mode=0o700, parents=True, exist_ok=True)
        learning_rate_scheduler = LearningRateScheduler(build_step_decay(initial_alpha, factor), verbose=1)
        weights_path = Path(output_path, 'weights.h5')
        checkpoint = ModelCheckpoint(str(weights_path), monitor="val_loss", save_best_only=True, verbose=1)
        
        print('##   Model Summary')
        with open(Path(output_path, 'model_summary.txt'), 'w') as target:
            print_summary(target, model)
            
        if validation_split == None:
            assert(not (validation_X is None))
            assert(not (validation_Y is None))
            validation_data = (validation_X, validation_y)
        else:
            assert((validation_X is None))
            assert((validation_Y is None))
            validation_data = None

        print("##   Training network...")
        history = model.fit(train_X, train_y, 
                            batch_size=self.train_batch_size,
                            epochs=self.train_epochs, 
                            validation_split=validation_split,
                            validation_data=validation_data,
                            verbose=self.verbose,
                            shuffle=True,
                            callbacks=[learning_rate_scheduler, checkpoint])

        print("##   Evaluating network...")
        predictions = model.predict(test_X, batch_size=self.predict_batch_size)
        print_classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), labels)
        plot_confusion_matrix(test_y.argmax(axis=1), predictions.argmax(axis=1))
        confusion_matrix_path = str(Path(output_path, 'confusion_matrix.png'))
        plt.savefig(confusion_matrix_path)
        plot_history_loss_and_accuracy(history)
        loss_and_accuracy_path = str(Path(output_path, 'loss_and_accuracy.png'))
        plt.savefig(loss_and_accuracy_path)
        print_history_keys(history)
        print("##   Done")

