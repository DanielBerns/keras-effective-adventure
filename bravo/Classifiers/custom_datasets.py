from typing import Dict, List, Tuple

from .sources import ImageWithLabelSource
from .processors import ResizeImageProcessor, ImageToArrayProcessor, SmallDatasetOfImages
from .loaders import Loader


def create_custom_image_dataset(expected_shape: Tuple[int, int, int], base_paths: List[str]) -> SmallDatasetOfImages:
    """This function builds a SmallDatasetOfImages.
          Using this SmallDatasetOfImages you can write
              trainX, trainY, testX, testY, data_shape, set_of_labels = dataset.get()
          or
              images = dataset.images
              labels = dataset.labels
    """
    source = ImageWithLabelSource(base_paths)
    dataset = SmallDatasetOfImages(expected_shape=expected_shape)
    processors = [ResizeImageProcessor(expected_shape), ImageToArrayProcessor()]
    loader = Loader(source, processors, dataset)
    loader.execute()
    dataset.build()
    return dataset

    

