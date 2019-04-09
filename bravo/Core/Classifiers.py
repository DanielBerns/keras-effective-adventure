# import the necessary packages
from pathlib import Path


from keras.models.mnist import load_model
from keras.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt


def print_classification_report(y_true, y_predicted, labelNames):
    print("Classification Report")
    print(classification_report(y_true, y_predicted, target_names=labelNames))
    value = accuracy_score(y_true, y_predicted)
    print(f"Accuracy : {str(value)}")


def plot_confusion_matrix(y_true, y_predicted, labelNames):
    mtx = confusion_matrix(y_true, y_predicted)
    fig, ax = plt.subplots(figsize=(8,8))
    sns.heatmap(mtx, annot=True, fmt='d', linewidths=.5,  cbar=False, ax=ax)
    plt.ylabel('true label')
    plt.xlabel('predicted label')


def plot_history_loss_and_accuracy(history):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
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


def train(model, train_X, train_y, test_X, test_y, labels, parameters):
    # train the network
    print("[INFO] training network...")
    history = model.fit(train_X, train_y, 
                        validation_split=parameters["validation_split"]
                        batch_size=parameters["batch_size"], 
                        epochs=parameters["epochs"], 
                        verbose=parameters["verbose"])
    # test the network
    print("[INFO] evaluating network...")
    output_path = Path(parameters["output"])
    predictions = model.predict(test_X, batch_size=parameters["batch_size"])
    
    confusion_matrix = metrics.confusion_matrix(test_y.argmax(axis=1), predictions.argmax(axis=1))
    print_classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), labels)
    plot_confusion_matrix(test_y.argmax(axis=1), predictions.argmax(axis=1), labels)
    confusion_matrix_path = str(Path(output_path, 'confusion_matrix.png'))
    plt.savefig(confusion_matrix_path)
    plot_history_loss_and_accuracy(history)
    loss_and_accuracy_path = str(Path(output_path, 'loss_and_accuracy.png'))
    plt.savefig(loss_and_accuracy_path)
    print("[INFO] Done")
    return history

