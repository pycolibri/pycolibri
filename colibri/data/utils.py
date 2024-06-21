import os

import torchvision
from PIL import Image

BUILTIN_DATASETS = {
    'mnist': torchvision.datasets.MNIST,
    'fashion_mnist': torchvision.datasets.FashionMNIST,
    'cifar10': torchvision.datasets.CIFAR10,
    'cifar100': torchvision.datasets.CIFAR100
}


def load_builtin(path, **kwargs):
    name = kwargs['name']
    train = kwargs['train'] if 'train' in kwargs else True
    download = kwargs['download'] if 'download' in kwargs else True

    return BUILTIN_DATASETS[name](root=path, train=train, download=download)


def load_img(filename, **kwargs):
    return Image.open(filename)


def load_mat(filename, preprocessing, **kwargs):
    pass


def load_h5(path, preprocessing, **kwargs):
    pass
