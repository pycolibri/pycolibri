import os
from PIL import Image
import scipy.io as sio

import torchvision


BUILTIN_DATASETS = {
    'mnist': torchvision.datasets.MNIST,
    'fashion_mnist': torchvision.datasets.FashionMNIST,
    'cifar10': torchvision.datasets.CIFAR10,
    'cifar100': torchvision.datasets.CIFAR100
}


def get_name_from_key(key):
    name = key
    if 'input' in key:
        name = 'input'
    elif 'output' in key:
        name = 'output'

    return name


def get_filenames(path, extension, **kwargs):
    if extension == 'builtin':
        name = kwargs['name']
        path = os.path.join(path, name)
        os.makedirs(path, exist_ok=True)

        return [path]

    else:
        """Return a list with the filenames of the files in the given path."""
        return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def load_builtin_dataset(path, **kwargs):
    name = kwargs['name']
    train = kwargs['train'] if 'train' in kwargs else True
    download = kwargs['download'] if 'download' in kwargs else True

    return BUILTIN_DATASETS[name](root=path, train=train, download=download)


def load_img(filename, **kwargs):
    return Image.open(filename)


def load_mat(filename, preprocessing, **kwargs):
    return sio.loadmat(filename)


def load_h5(path, preprocessing, **kwargs):
    pass
