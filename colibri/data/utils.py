import torchvision

BUILTIN_DATASETS = {
    'mnist': torchvision.datasets.MNIST,
    'fashion_mnist': torchvision.datasets.FashionMNIST,
    'cifar10': torchvision.datasets.CIFAR10,
    'cifar100': torchvision.datasets.CIFAR100
}


def load_builtin_dataset(path, preprocessing=None, transforms=None):
    train_dataset = BUILTIN_DATASETS[path](root='data', train=True, download=True, transform=preprocessing)
    test_dataset = BUILTIN_DATASETS[path](root='data', train=False, download=True, transform=preprocessing)

    return train_dataset, test_dataset


def load_img_dataset(path, preprocessing):
    pass


def load_mat_dataset(path, preprocessing):
    pass


def load_h5_dataset(path, preprocessing):
    pass
