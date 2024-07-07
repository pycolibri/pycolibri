import matplotlib.pyplot as plt
import torch

from torch.utils import data
from torch.utils.data import Dataset
from torchvision import transforms

import colibri.data.utils as D


DATASET_READER = {
    'cave': D.load_cave_sample,
    'arad': D.load_arad_sample,
}


class DefaultTransform:
    """Default transformation class.

    This class is used to apply the default transformations to the data.

    The default transformations are:
        - input: `transforms.ToTensor()`
        - output: `transforms.ToTensor()`

    The default transformation for the output is `transforms.ToTensor()`, but it can be changed by providing a dictionary with the key 'output' in the `transform_dict` parameter.
    The default transformation for the input is `transforms.ToTensor()`, but it can be changed by providing a dictionary with the key 'input' in the `transform_dict` parameter.

    """

    def __init__(self, name):
        """
        Arguments:
            name (string): Name of the dataset.
        """
        self.transform_dict = dict(input=transforms.ToTensor(), default=transforms.ToTensor())
        if name in D.BUILTIN_DATASETS:
            self.transform_dict['output'] = transforms.Lambda(lambda x: x)
        else:
            self.transform_dict['output'] = transforms.ToTensor()

    def __call__(self, key, value):
        """
        Arguments:
            key (string): Key of the data.
            value (object): Data to transform.

        Returns:
            object: Transformed data.
        """
        if key in self.transform_dict:
            return self.transform_dict[key](value)
        else:
            return self.default_transform(value)

    def default_transform(self, data):
        """
        Arguments:
            data (object): Data to transform

        Returns:
            object: Transformed data.
        """
        return self.transform_dict['default'](data)


class CustomDataset(Dataset):
    """Custom dataset.

    This class allows to load custom datasets and apply transformations to the data.

    The datasets that can be currently loaded are:
        - 'cifar10'
        - 'mnist'
        - 'fashion_mnist'
        - 'cave'

    This class is dividied in two parts:
        - builtin datasets: datasets that are predefined in the repository.
        - custom datasets: datasets that are not predefined in the repository.

    The builtin datasets are loaded using the function `load_builtin_dataset` from the module `colibri.data.utils`.
    The custom datasets are loaded using the function `get_filenames` from the module `colibri.data.utils`.
    The transformations are applied to the data using the `torchvision.transforms` module.

    The default transformations are:
        - input: `transforms.ToTensor()`
        - output: `transforms.ToTensor()`

    The default transformation for the output is `transforms.ToTensor()`, but it can be changed by providing a dictionary with the key 'output' in the `transform_dict` parameter.
    The default transformation for the input is `transforms.ToTensor()`, but it can be changed by providing a dictionary with the key 'input' in the `transform_dict` parameter.
    The `transform_dict` parameter is a dictionary with the transformations to apply to the data.

    Example:
        >>> from torchvision import transforms
        >>> name = 'cifar10'
        >>> path = '.'
        >>> builtin_dict = dict(train=True, download=True)
        >>> transform_dict = dict(input=transforms.ToTensor(), output=transforms.ToTensor())
        >>> dataset = CustomDataset(name, path, builtin_dict=builtin_dict, transform_dict=transform_dict)
    """
    def __init__(self, name, path, builtin_dict=None, transform_dict=None):
        """
        Arguments:
            name (string): Name of the dataset.
                Current options are: ('cifar10', 'cifar100', 'mnist', 'fashion_mnist', 'cave').
            path (string): Path to directory with the dataset.
            builtin_dict (dict): Dictionary with the parameters to load the builtin dataset.
            transform_dict (dict,object): Dictionary with the transformations to apply to the data.
        """
        if transform_dict is None:
            transform_dict = {}
        else:
            assert 'input' in transform_dict, "'input' key must be provided in transform_dict"

        self.is_builtin_dataset = False
        if name in D.BUILTIN_DATASETS:  # builtin datasets
            assert builtin_dict is not None, "builtin_dict must be provided for builtin datasets"
            self.is_builtin_dataset = True
            path = D.update_builtin_path(name, path)
            self.dataset = D.load_builtin_dataset(name, path, **builtin_dict)
            self.len_dataset = len(self.dataset['input'])

        else:  # custom datasets
            self.dataset_filenames = D.get_filenames(name, path)
            self.data_reader = DATASET_READER[name]
            self.len_dataset = len(self.dataset_filenames)

        self.transform_dict = transform_dict
        self.default_transform = DefaultTransform(name)

    def __len__(self):
        """
        Returns:
            int: Length of the dataset.
        """
        return self.len_dataset

    def __getitem__(self, idx):
        """
        Arguments:
            idx (int): Index of the sample to load.

        Returns:
            dict: Dictionary with the data.
        """

        # load sample

        if self.is_builtin_dataset:
            data = {key: value[idx] for key, value in self.dataset.items()}
        else:
            data = self.data_reader(self.dataset_filenames[idx])

        # apply transformation

        for key, value in data.items():
            if not isinstance(value, torch.Tensor):
                if key in self.transform_dict:
                    data[key] = self.transform_dict[key](value)
                else:
                    data[key] = self.default_transform(key, value)

        return data
