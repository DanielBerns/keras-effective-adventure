# USAGE
# python shallownet_animals.py --dataset animals

# import the necessary packages
import argparse

from sklearn.metrics import classification_report
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

from .Sources import ImageWithLabelSource
from .Processors import ResizeImageProcessor
from .Processors import ImageToArrayProcessor
from .Processors import ImageContext
from .Loaders import Loader

class Trainer:
    def __init__(self, context):       
        self._testX = None
        self._testY = None
        # construct the argument parse and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
        ap.add_argument("-m", "--model", required=True, help="path to output model")
        ap.add_argument("-t", "--test_size", required=False, type=int, default=25, 
                        help="partition the samples intro training and testing splits (10-50)")
        ap.add_argument("-b", "--batch_size", required=False, type=int, default=32, help="batch size")
        ap.add_argument("-e", "--epochs", required=False, type=int, default=100, help="training epochs")
        self._args = vars(ap.parse_args())
        # grab the list of images that we'll be describing
        print("[INFO] loading images...")
        # initialize the image preprocessor, load the dataset from disk,
        # and reshape the samples matrix
        source = ImageWithLabelSource(self._args["dataset"])
        # image_to_array = ImageToArrayProcessor()
        resize_image = ResizeImageProcessor(32, 32)
        self._context = ImageContext(expected_shape=(32, 32, 3), verbose=500)
        loader = Loader(source, self.context, [resize_image])
        loader.execute()

    @property
    def testX(self):
        return self._testX

    @testX.setter
    def testX(self, value):
        self._testX = value 
    
    @property
    def testY(self):
        return self._textY

    @testY.setter
    def testY(self, value):
        self._testY = value 
    
    @property
    def args(self):
        return self._args
    
    @property
    def context(self):
        return self._context

    def execute(self, model):
        # partition the data into training and testing splits
        (trainX, trainY, testX, testY) = self.context.get_dataset(float(self.args["test_size"])/100.0)

        self.testX = testX
        self.testY = testY

        # train the network
        print("[INFO] training network...")
        H = model.fit(trainX, trainY, validation_data=(testX, testY), 
                      batch_size=self.args["batch_size"], epochs=self.args["epochs"], 
                      verbose=1)
        
        # evaluate the network
        print("[INFO] evaluating network...")
        predictions = model.predict(testX, batch_size=self.args["batch_size"])
        print(classification_report(
            testY.argmax(axis=1), 
            predictions.argmax(axis=1), 
            target_names=list(self.context.encoder.classes_)))
        
        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
        plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.show()

    def verify(self, model, number_of_samples=-1):
        label_encoder = self.context.label_encoder
        images_list, true_labels_list = self.get_random_images_and_labels(number_of_samples)
        
        # evaluate the network
        print("[INFO] predicting...")
        scores = model.predict(images_list, batch_size=32)
        predicted_labels_list = scores.argmax(axis=1)
        
        true_names_list = self.context.encoder.inverse_transform(true_labels_list)
        predicted_names_list = self.context.encoder.inverse_transform(predicted_labels_list)
        # loop over the sample images
        for k, image, true_name, predicted_name in zip(index, images_list, true_names_list, predicted_name_list):
            fig = plt.figure()
            plt.imshow(image)
            plt.title('Sample {0:d} - True label {1:s} - prediction {2:s}'.format(
                      k, true_name, predicted_name))
            plt.show()
