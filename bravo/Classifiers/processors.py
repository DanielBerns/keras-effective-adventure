# import the necessary packages
from skimage.transform import resize
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
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
        self._set_initial_values()
        
    def _set_initial_values(self):
        self._current_image = None
        self._current_label = None
        self._images_list = []
        self._labels_list = []
        self._images = None
        self._labels = None
        self._codes = None
        self._onehots = None
        self._set_of_labels = None

    @property
    def expected_shape(self):
        return self._expected_shape
    
    @property
    def verbose(self):
        return self._verbose

    @property
    def current_image(self):
        return self._current_image
    
    @current_image.setter
    def current_image(self, value):
        self._current_image = value

    @property
    def current_label(self):
        return self._current_label
    
    @current_label.setter
    def current_label(self, value):
        self._current_label = value

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def codes(self):
        return self._codes
    
    @property
    def onehots(self):
        return self._onehots

    @property
    def set_of_labels(self):
        return self._set_of_labels

    def get_label(self, onehot):
        if len(onehot) == len(self.set_of_labels):
            return self.set_of_labels[onehot.argmax()]
        else:
            return "onehot.error"
        
    def add_sample(self):
        if self.current_image is None:
            raise ContextException('ImageContext.add_sample: unexpected current_image is None')
        try:
            if self.current_image.shape != self.expected_shape:
                raise ContextException('ImageContext.add_sample: unexpected image shape')
        except AttributeError:
            raise ContextException('ImageContext.add_sample: unexpected current_image has no shape attribute')            
        self._images_list.append(self.current_image)
        self._labels_list.append(self.current_label)
        if self.verbose > 0:
            count = len(self._images_list) + 1
            if (count % self.verbose) == 0:
                print(f'{count:>5d} samples processed.')

    def start(self):
        self._set_initial_values()

    def stop(self):
        pass

    def build_images_and_labels(self):
        images = np.array(self._images_list)
        set_of_labels, codes = np.unique(self._labels_list, return_inverse=True)
        onehots = np_utils.to_categorical(codes)
        self._images = images
        self._labels = set_of_labels[onehots.argmax(1)]
        self._codes = codes
        self._onehots = onehots
        self._set_of_labels = set_of_labels

    def get_dataset(self, branch=0.20, random_state=42):
        np.random.seed(random_state)
        train_X = list()
        train_Y = list()
        validation_X = list()
        validation_Y = list()
        test_X = list()
        test_Y = list()
        train_branch = 1.0 - branch
        validation_branch = train_branch**2
        for image, onehot in zip(self.images, self.onehots):
            die = np.random.uniform()
            if die < validation_branch:
                train_X.append(image)
                train_Y.append(onehot)
            elif die < train_branch:
                validation_X.append(image)
                validation_Y.append(onehot)
            else:
                test_X.append(image)
                test_Y.append(onehot)
        train_X = np.array(train_X)
        train_Y = np.array(train_Y)
        validation_X = np.array(validation_X)
        validation_Y = np.array(validation_Y)
        test_X = np.array(test_X)
        test_Y = np.array(test_Y)
        return train_X, train_Y, validation_X, validation_Y, test_X, test_Y
                           
        
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
    def __init__(self, shape):
        # the target image width and height for resizing
        super().__init__()
        self._height = shape[0]
        self._width = shape[1]

    @property
    def height(self):    
        return self._height

    @property
    def width(self):
        return self._width

    def execute(self):
        # resize the image to a fixed size, ignoring the aspect ratio
        self.context.current_image = \
            resize(self.context.current_image, 
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
        self.context.current_image = img_to_array(self.context.current_image)

