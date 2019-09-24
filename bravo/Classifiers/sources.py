import os
from pathlib import Path

from skimage.io import imread


def list_files(base, valid_extensions):
    # loop over the directory structure
    if '~' in base:
        base_path = Path(base).expanduser()
    else:
        base_path = Path(base)
    base_path = base_path.absolute()
    for (root, directories, filenames) in os.walk(base_path):
        # loop over the filenames in the current directory
        for filename in filenames:
            # determine the file extension of the current file
            extension = filename[filename.rfind("."):].lower()
            # check to see if the file should be processed
            if extension.endswith(valid_extensions):
                # construct the file path
                file_path = Path(root, filename)
                yield root, file_path


def read_file_path_with_label(base_path, valid_extensions):
    label_length = len(base_path)
    for root, file_path in list_files(base_path, valid_extensions):
        label = root[label_length:]
        yield (file_path, label)



class ImageWithLabelSource:
    def __init__(self, base_paths=None, valid_extensions=None):
        if base_paths == None:
            base_paths = [Path('./')]
        self._base_paths = base_paths
        if valid_extensions == None:
            valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        self._valid_extensions = valid_extensions

    @property
    def base_paths(self):
        return self._base_paths
    
    @property
    def valid_extensions(self):
        return self._valid_extensions

    def read(self):
        for base_path in self.base_paths:
            for (file_path, label) in read_file_path_with_label(base_path, self.valid_extensions):
                image = imread(str(file_path))
                yield (image, label)
    
