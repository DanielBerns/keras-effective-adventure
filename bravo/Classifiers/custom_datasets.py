from typing import Dict, List, Tuple

from .sources import ImageWithLabelSource
from .processors import ResizeImageProcessor, ImageToArrayProcessor, ImageContext
from .loaders import Loader


def get_custom_image_context(expected_shape: Tuple[int, int, int], base_paths: List[str]) -> ImageContext:
    """This function builds a custom ImageContext.
          Using this ImageContext you can write
              trainX, trainY, testX, testY = context.get_dataset()
          or
              images = context.images
              labels = context.labels
              encoder = context.encoder
    """
    source = ImageWithLabelSource(base_paths)
    context = ImageContext(expected_shape=expected_shape)
    processors = [ResizeImageProcessor(expected_shape), ImageToArrayProcessor()]
    loader = Loader(source, context, processors)
    loader.execute()
    context.build_images_and_labels()
    return context

    

