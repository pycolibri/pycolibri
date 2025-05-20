import os

import numpy as np
from PIL import Image

import torchvision
import torch
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
    dataset['input'] = np.transpose(dataset['input'], (0, 3, 2, 1))

    dataset['input'] = torch.from_numpy(dataset['input'])
    dataset['output'] = torch.tensor(dataset['output'])
    return dataset

