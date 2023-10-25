import pytest

@pytest.fixture
def dataset():
    
    dataset_path = 'cifar10'
    keys = ''
    batch_size = 32
    return Dataset(dataset_path, keys=keys, batch_size=batch_size)

def test_data_size(dataset):

    sample = next(iter(dataset.train_dataset))
    expected_size = (dataset.batch_size, 3, 32, 32)  

    assert sample[0].shape == expected_size

def test_data_output_type(dataset):

    sample = next(iter(dataset.train_dataset))
    expected_type = tuple
    
    assert isinstance(sample, expected_type)