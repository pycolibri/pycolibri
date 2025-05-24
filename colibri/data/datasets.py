import torch
from torch.utils.data import Dataset
from torchvision import transforms

from colibri.data.sota_datasets import CaveDataset
from colibri.data.utils import BUILTIN_DATASETS, update_builtin_path, load_builtin_dataset

DATASET_HANDLERS = {
    'cave': CaveDataset,
}


class DefaultTransform:
    r"""Default transformation class.

    This class is used to apply the default transformations to the data.

    The default transformations are:
        - input: `transforms.ToTensor()`
        - output: `transforms.ToTensor()`

    The default transformation for the output is `transforms.ToTensor()`, but it can be changed by providing a dictionary with the key 'output' in the `transform_dict` parameter.
    The default transformation for the input is `transforms.ToTensor()`, but it can be changed by providing a dictionary with the key 'input' in the `transform_dict` parameter.

    """

    def __init__(self, name):
        r"""
        Args:
            name (string): Name of the dataset.
        """
        self.transform_dict = dict(input=transforms.ToTensor(), default=transforms.ToTensor())
        if name in BUILTIN_DATASETS:
            self.transform_dict['output'] = transforms.Lambda(lambda x: x)
        else:
            self.transform_dict['output'] = transforms.ToTensor()

    def __call__(self, key, value):
        r"""
        Args:
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
        r"""
        Args:
            data (object): Data to transform

        Returns:
            object: Transformed data.
        """
        return self.transform_dict['default'](data)


class CustomDataset(Dataset):
    r"""Custom dataset.

    This class allows to load custom datasets and apply transformations to the data.

    The datasets that can be currently loaded are:
        - 'cifar10'
        - 'mnist'
        - 'fashion_mnist'
        - 'cave'

    This class is divided in two parts:
        - builtin datasets: datasets that are predefined in the repository.
        - custom datasets: datasets that are not predefined in the repository.

    The builtin datasets are loaded using the function `load_builtin_dataset` from the module `colibri.data.utils`.
    The custom datasets are loaded using the function `get_filenames` from the module `colibri.data.utils`.
    The transformations are applied to the data using the `torchvision.transforms` module.

    The default transformations are:
        - input: `transforms.ToTensor()`
        - output: `transforms.ToTensor()`

    """
    def __init__(self, name: str, path: str= "data", builtin_train: bool=True, builtin_download: bool=True, transform_dict:dict={}):
        r"""
        Args:
            name (string): Name of the dataset.
                Current options are: ('cifar10', 'cifar100', 'mnist', 'fashion_mnist', 'cave').
            path (string): Path to directory with the dataset.
            builtin_train (bool): Whether to load the training or test set. This option is only available for builtin datasets.
            builtin_download (bool): Whether to download the dataset if it is not found. This option is only available for builtin datasets.
            transform_dict (dict,object): Dictionary with the transformations to apply to the data.
        """

        self.is_builtin_dataset = False
        if name in BUILTIN_DATASETS:  # builtin datasets
            # assert kwargs_builtin is not None, "kwargs_builtin must be provided for builtin datasets"
            self.is_builtin_dataset = True
            path = update_builtin_path(name, path)
            self.dataset = load_builtin_dataset(name, path, builtin_train, builtin_download)
            self.len_dataset = len(self.dataset['input'])

        else:  # custom datasets
            self.data_handler = DATASET_HANDLERS[name](path)
            self.dataset_filenames = self.data_handler.get_list_paths()
            self.data_reader = self.data_handler.load_item
            self.len_dataset = len(self.dataset_filenames)

        self.transform_dict = transform_dict

    def __len__(self):
        r"""
        Returns:
            int: Length of the dataset.
        """
        return self.len_dataset

    def __getitem__(self, idx):
        r"""
        Args:
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

        for key, value in self.transform_dict.items():
            data[key] = self.transform_dict[key](value)

        return data