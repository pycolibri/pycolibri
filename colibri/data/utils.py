import os
from PIL import Image

import numpy as np


def get_all_filenames(root_directory):
    """
    Get a list of file paths for image files in the specified directory.

    Args:
        root_directory (str): The root directory to search for image files.

    Returns:
        list: A list of file paths for image files in the directory, sorted.
    """
    file_paths = []

    # Walk through the directory tree using os.walk
    for root, _, files in os.walk(root_directory):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.mat')):
                # Get the full path of the file
                file_path = os.path.join(root, filename)
                file_paths.append(file_path)

    file_paths.sort()
    return file_paths


def load_image(filename):
    """
    Load a jpg, jpeg or png image

    Args:
        filename (str): The filename to be loaded.

    Returns:
        image (array): The normalized image [0, 1] as numpy array.
    """
    image = Image.open(filename)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return np.array(image) / 255.
