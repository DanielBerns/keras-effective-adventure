import os
import random


from skimage.io import imread


class FilePathWithLabelSource:
    def __init__(self, base_path, valid_extensions):
        self._base_path = base_path
        self._valid_extensions = valid_extensions
    
    @property
    def base_path(self):
        return self._base_path
    
    def read(self):
        for root, file_path in list_files(self.base_path, valid_extensions=self._valid_extensions):
            label = root[len(self.base_path):]
            yield (file_path, label)


class ImageWithLabelSource(FilePathWithLabelSource):
    def __init__(self, base_path):
        super().__init__(base_path, (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    
    def read(self):
        for (file_path, label) in super().read():
            image = imread(file_path)
            yield (image, label)


def list_files(base_path, valid_extensions):
    # loop over the directory structure
    for (root, directories, filenames) in os.walk(base_path):
        # loop over the filenames in the current directory
        for filename in filenames:
            # determine the file extension of the current file
            extension = filename[filename.rfind("."):].lower()
            # check to see if the file should be processed
            if extension.endswith(valid_extensions):
                # construct the file path
                file_path = os.path.join(root, filename)
                yield root, file_path
