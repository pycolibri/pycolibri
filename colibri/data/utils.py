import os

import numpy as np
from PIL import Image

import torchvision

# Builtin datasets

BUILTIN_DATASETS = {

    'mnist': torchvision.datasets.MNIST,
    'fashion_mnist': torchvision.datasets.FashionMNIST,
    'cifar10': torchvision.datasets.CIFAR10,
    'cifar100': torchvision.datasets.CIFAR100
}


def update_builtin_path(name: str, path: str):
    r"""
    Update the built-in path by creating a new directory with the given name
    inside the specified path.
    Args:
        name (str): The name of the directory to be created.
        path (str): The path where the new directory will be created.
    Returns:
        str: The path of the newly created directory.
    """

    path = os.path.join(path, name)
    os.makedirs(path, exist_ok=True)

    return path


def load_builtin_dataset(name: str, path: str, **kwargs):
    r"""
    Load a built-in dataset.
    Args:
        name (str): The name of the dataset.
        path (str): The path to save the dataset.
        **kwargs: Additional keyword arguments to pass to the pytorch dataset loader.

    Returns:
        dict: A dictionary containing the input and output data of the dataset.

    Raises:
        KeyError: If the specified dataset name is not found.
        
    """

    train = kwargs['train'] if 'train' in kwargs else False
    download = kwargs['download'] if 'download' in kwargs else True

    builtin_dataset = BUILTIN_DATASETS[name](root=path, train=train, download=download)
    dataset = dict(input=builtin_dataset.data, output=builtin_dataset.targets)

    # transform

    dataset['input'] = (dataset['input'] / 255.).astype(np.float32)
    if dataset['input'].ndim != 4:
        dataset['input'] = dataset['input'].unsqueeze(1)

    return dataset


# Custom datasets

def get_cave_filenames(path:str):
    r"""
    Returns a list of cave filenames in the given path.

    Args:
        path (str): The path to the directory containing the cave files.
    Returns:
        list: A list of cave filenames.
    """

    return [os.path.join(path, name, name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]


def get_arad_filenames(path:str):
    pass


CUSTOM_DATASETS = {
    'cave': get_cave_filenames,
    'arad': get_arad_filenames
}


def get_filenames(name : str, path : str):
    r"""
    Get the filenames of the custom dataset.
    Args:
        name (str): The name of the dataset.
        path (str): The path to the directory containing the dataset.
    Returns:
        list: A list of filenames.
    Raises:
        KeyError: If the specified dataset name is not found.
    """

    return CUSTOM_DATASETS[name](path)


def load_arad_sample(filename: str, preprocessing, **kwargs):
    r"""
    
    Load a sample from the ARAD dataset.
    Args:
        filename (str): The filename of the sample.
        preprocessing (function): The preprocessing function to apply to the data.
        **kwargs: Additional keyword arguments to pass to the preprocessing function.
    Returns:
        dict: A dictionary containing the input and output data of the sample.
    
    """
    return Image.open(filename)


def load_cave_sample(filename: str):
    r"""
    Load a sample from the CAVE dataset.
    Args:
        filename (str): The filename of the sample.
    Returns:
        dict: A dictionary containing the input and output data of the sample.
    """
    name = os.path.basename(filename).replace('_ms', '')

    spectral_image = []
    for i in range(1, 32):
        spectral_band_filename = os.path.join(filename, f'{name}_ms_{i:02d}.png')
        spectral_band = np.array(Image.open(spectral_band_filename))
        spectral_band = spectral_band / (2 ** 16 - 1) if isinstance(spectral_band[0, 0], np.uint16) else spectral_band
        spectral_band = spectral_band / (2 ** 8 - 1) if isinstance(spectral_band[0, 0], np.uint8) else spectral_band
        spectral_image.append(spectral_band.astype(np.float32))
    spectral_image = np.stack(spectral_image, axis=-1)

    rgb_image = np.array(Image.open(os.path.join(filename, f'{name}_RGB.bmp'))) / 255.
    rgb_image = rgb_image.astype(np.float32)

    return dict(input=rgb_image, output=spectral_image)
