# USAGE
# python shallownet_animals.py --dataset animals

# import the necessary packages
from pathlib import Path


from keras.models import load_model
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


def plot_history_loss_and_accuracy(history_keras):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(history_keras.history['loss'])
    axs[0].plot(history_keras.history['val_loss'])
    axs[0].set_title('model loss')
    axs[0].set_ylabel('loss')
    axs[0].set_xlabel('epoch')
    axs[0].legend(['train', 'validation'], loc='upper left')
    axs[1].plot(history_keras.history['acc'])
    axs[1].plot(history_keras.history['val_acc'])
    axs[1].set_title('model accuracy')
    axs[1].set_ylabel('accuracy')
    axs[1].set_xlabel('epoch')
    axs[1].legend(['train', 'validation'], loc='upper left')


class Dataset:
    def __init__(self):
        pass
    
    def train_data(self):
        return (None, None)
    
    def validation_data(self):
        return (None, None)
    
    def test_data(self):
        return (None, None)
    
    def labels_data(self):
        return (None, None)

    
class DatasetBuilder:
    def __init__(self):
        self._dataset = None
    
    @property
    def dataset(self):
        return self._dataset
    
    @dataset.setter
    def dataset(self, value):
        self._dataset = value


class ModelBuilder:
    def __init__(self):
        self._model = None
    
    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, value):
        self._model = value
    


def train(dataset_builder, model_builder, parameters):
    # build dataset
    print("[INFO] building dataset...")
    dataset = dataset_builder.dataset(parameters)
    train_X, train_y = dataset.train_data()
    validation_X, validation_y = dataset.validation_data()
    test_X, test_y = dataset.test_data()
    labels, label_encoder = dataset.labels_data()
    
    # build model
    print("[INFO] building model...")
    model = model_builder.model(parameters)

    # train the network
    print("[INFO] training network...")
    history = model.fit(train_X, train_y, 
                        validation_data=(validation_X, validation_y), 
                        batch_size=parameters["batch_size"], 
                        epochs=parameters["epochs"], 
                        verbose=parameters["verbose"])

    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(test_X, batch_size=32)
    print_classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), labels)
    plot_confusion_matrix(test_y.argmax(axis=1), predictions.argmax(axis=1), labels)
    confusion_matrix_path = str(Path(output_path, 'confusion_matrix.png'))
    plt.savefig(confusion_matrix_path)
    plot_history_loss_and_accuracy(history)
    loss_and_accuracy_path = str(Path(output_path, 'loss_and_accuracy.png'))
    plt.savefig(loss_and_accuracy_path)
    print("[INFO] Done")
    return model, labels, labels_encoder


# def verify(model, labels_encoder, images_list, true_labels_list):
#     # evaluate the network
#     print("[INFO] predicting...")
#     scores = model.predict(images_list, batch_size=32)
#     predicted_labels_list = scores.argmax(axis=1)
#     true_names_list = labels_encoder.inverse_transform(true_labels_list)
#     predicted_names_list = labels_encoder.inverse_transform(predicted_labels_list)
    
