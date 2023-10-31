import pytest
from .utils import include_colibri

include_colibri()

from colibri_hdsp.data.datasets import Dataset


@pytest.fixture
def dataset():
    dataset_path = 'cifar10'
    keys = ''
    batch_size = 32
    return batch_size, Dataset(dataset_path, keys=keys, batch_size=batch_size)


def test_data_size(dataset):
    batch_size, data = dataset
    sample = next(iter(data.train_dataset))
    expected_size = (batch_size, 3, 32, 32)

    assert sample[0].shape == expected_size


def test_data_output_type(dataset):
    _, data = dataset
    sample = next(iter(data.train_dataset))
    expected_type_0 = tuple
    expected_type_1 = list

    assert isinstance(sample, expected_type_0) or isinstance(sample, expected_type_1)
