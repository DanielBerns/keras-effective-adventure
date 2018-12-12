# import the necessary packages
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import img_to_array
import numpy as np

#----------------------------------------------------------------------
class ContextException(Exception):
    def __init__(self, message):
        super().__init__(message)


#----------------------------------------------------------------------
class ImageContext:
    def __init__(self, expected_shape, verbose):
        self._expected_shape = expected_shape
        self._verbose = verbose
        self._count = -1
        self._filepath = None
        self._image = None
        self._label = None
        self._filepaths_list = None
        self._label_encoder = LabelEncoder()
        self._samples_list = None
        self._labels_list = None

    @property
    def count(self):
        return self._count
    
    @count.setter
    def count(self, value):
        self._count = value

    @property
    def verbose(self):
        return self._verbose

    @property
    def filepath(self):
        return self._filepath
    
    @filepath.setter
    def filepath(self, value):
        self._filepath = value
   
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
        
    def add_sample_and_label(self):
        if self.image.shape != self._expected_shape:
            raise ContextException('ImageContext.add_sample_and_label: unexpected image shape')
        self._filepaths_list.append(self.filepath)
        self._samples_list.append(self.image)
        self._labels_list.append(self.label)
        if self._verbose > 0:
            if (self.count % self.verbose) == 0:
                print('{0:>5d} samples processed.'.format(self.count + 1))

    def start(self):
        self._filepaths_list = []
        self._samples_list = []
        self._labels_list = []
    
    def stop(self):
        pass

    def label_encoder(self):
        return self._label_encoder
        
    def samples_and_labels(self):
        samples = np.array(self._samples_list)
        labels = self._label_encoder.fit_transform(self._labels_list)
        return samples, labels
                           
    @property
    def filepaths(self):
        return self._filepaths_list

#----------------------------------------------------------------------
class ImageAsVectorContext(ImageContext):
    def __init__(self, expected_shape=(224, 224, 3), verbose=-1):
        super().__init__(expected_shape, verbose)
   
    def samples_and_labels(self):
        _samples, labels = super().samples_and_labels()
        shape = _samples.shape
        samples = _samples.reshape((shape[0], shape[1] * shape[2] * 3))
        return samples, labels

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
    def __init__(self, data_format=None):
        # store the image data format
        super().__init__()
        self._data_format = data_format

    @property
    def data_format(self):    
        return self._data_format

    def execute(self):
        # apply the Keras utility function that correctly rearranges
        # the dimensions of the image
        self.context.image = img_to_array(self.context.image, data_format=self.data_format)

