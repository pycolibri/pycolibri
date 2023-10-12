import os
from PIL import Image

import numpy as np
import tensorflow as tf


def normalize_img(image, max_val=255.):
    """
    Normalize the input image.

    Args:
        image (tf.Tensor): The image to normalize.
        max_val (float, optional): The maximum value of the image (default is 255.0).

    Returns:
        tf.Tensor: The normalized image.
    """
    norm_img = tf.cast(image, tf.float32) / max_val
    return norm_img


def hot_encode(label, num_classes=10):
    """
    One-hot encode the label.

    Args:
        label (int): The label to encode.
        num_classes (int, optional): The number of classes (default is 10).

    Returns:
        tf.Tensor: The encoded label.
    """
    return tf.one_hot(label, num_classes)


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
    image = Image.open(filename)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return np.array(image) / 255.
