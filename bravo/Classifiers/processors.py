# import the necessary packages
from skimage.transform import resize
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
import numpy as np

#----------------------------------------------------------------------
class ContextException(Exception):
    def __init__(self, message):
        super().__init__(message)


#----------------------------------------------------------------------
class ImageContext:
    def __init__(self, expected_shape=(32, 32, 3), verbose=500):
        self._expected_shape = expected_shape
        self._verbose = verbose
        self._image = None
        self._label = None
        self._images_list = None
        self._labels_list = None
        self._encoder = None

    @property
    def expected_shape(self):
        return self._expected_shape
    
    @property
    def verbose(self):
        return self._verbose

    @property
    def image(self):
        return self._image
    
    @image.setter
    def image(self, value):
        self._image = value

    @property
    def label(self):
        return self._label
    
    @label.setter
    def label(self, value):
        self._label = value
        
    def add_sample(self):
        if self.image.shape != self.expected_shape:
            raise ContextException('ImageContext.add_sample: unexpected image shape')
        self._images_list.append(self.image)
        self._labels_list.append(self.label)
        if self._verbose > 0:
            count = len(self._images_list) + 1
            if (count % self.verbose) == 0:
                print(f'{count:>5d} samples processed.')

    def start(self):
        self._images_list = []
        self._labels_list = []
    
    def stop(self):
        pass

    def get_dataset(self, test_size=0.20):
        images = np.array(self._images_list)
        encoder = LabelBinarizer().fit(self._labels_list)
        labels = encoder.transform(self._labels_list)
        (trainX, testX, trainY, testY) = train_test_split(
            images, labels, 
            test_size=test_size, 
            random_state=42)
        return trainX, trainY, testX, testY, images, labels, encoder
                           

#----------------------------------------------------------------------
class Processor:
    def __init__(self):
        self._context = None
        
    @property
    def context(self):
        return self._context

    @context.setter
    def context(self, value):
        assert(self._context == None)
        self._context = value

    def start(self):
        pass
    
    def stop(self):
        pass


#----------------------------------------------------------------------
class ResizeImageProcessor(Processor):
    def __init__(self, width, height):
        # the target image width and height for resizing
        super().__init__()
        self._width = width
        self._height = height

    @property
    def width(self):
        return self._width

    @property
    def height(self):    
        return self._height

    def execute(self):
        # resize the image to a fixed size, ignoring the aspect ratio
        self.context.image = \
            resize(self.context.image, 
                   (self.width, self.height), anti_aliasing=True)


#----------------------------------------------------------------------
class ImageToArrayProcessor(Processor):
    def __init__(self):
        """PIL image to numpy array
        """
        super().__init__()

    def execute(self):
        # apply the Keras utility function that correctly rearranges
        # the dimensions of the image
        self.context.image = img_to_array(self.context.image)

