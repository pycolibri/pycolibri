import os

import torchvision
from torchvision import transforms
from PIL import Image

BUILTIN_DATASETS = {
    'mnist': torchvision.datasets.MNIST,
    'fashion_mnist': torchvision.datasets.FashionMNIST,
    'cifar10': torchvision.datasets.CIFAR10,
    'cifar100': torchvision.datasets.CIFAR100
}


class DefaultTransform:
    def __init__(self, extension):
        if extension == 'builtin':
            self.transform_data = transforms.ToTensor()
            self.transform_label = transforms.Lambda(lambda x: x)

        else:
            raise NotImplementedError

    def transform_data(self, data):
        return self.transform_data(data)

    def transform_label(self, label):
        return self.transform_label(label)


def get_filenames(path, extension, **kwargs):
    if extension == 'builtin':
        name = kwargs['name']
        path = os.path.join(path, name)
        os.makedirs(path, exist_ok=True)

        return [path]

    else:
        """Return a list with the filenames of the files in the given path."""
        return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

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
