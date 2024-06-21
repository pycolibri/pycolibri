import torchvision
from PIL import Image

BUILTIN_DATASETS = {
    'mnist': torchvision.datasets.MNIST,
    'fashion_mnist': torchvision.datasets.FashionMNIST,
    'cifar10': torchvision.datasets.CIFAR10,
    'cifar100': torchvision.datasets.CIFAR100
}


def load_builtin(path, train=True, download=True):
    return BUILTIN_DATASETS[path](root='data', train=train, download=download)


def load_img(filename):
    return Image.open(filename)


def load_mat(filename, preprocessing):
    pass


def load_h5(path, preprocessing):
    pass
