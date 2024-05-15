import pytest
from .utils import include_colibri
include_colibri()

import torch

from colibri.recovery.transforms import DCT2D

def load_img():

    from colibri.data.datasets import Dataset
    dataset_path = 'cifar10'
    keys = ''
    batch_size = 1
    dataset = Dataset(dataset_path, keys, batch_size)
    sample = next(iter(dataset.train_dataset))[0]
    return sample


def test_dct2d():

    x_true = load_img()
    transform_dct = DCT2D()

    theta = transform_dct.forward(x_true)
    x_hat = transform_dct.inverse(theta)

    assert torch.allclose(x_true, x_hat)

    
        