import os

from torch.utils import data
from torch.utils.data import Dataset

import colibri.data.utils as D

DATASET_READER = {
    'builtin': D.load_builtin,
    'img': D.load_img,
    'mat': D.load_mat,
    'h5': D.load_h5,
}


def get_filenames(path, extension, **kwargs):
    if extension == 'builtin':
        name = kwargs['name']
        path = os.path.join(path, name)
        os.makedirs(path, exist_ok=True)

        return [path]

    else:
        """Return a list with the filenames of the files in the given path."""
        return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


class CustomDataset(Dataset):
    """Custom dataset."""

    def __init__(self, path, extension, dataset_vars=None, transform_input=None, transform_output=None, preload=False):
        """
        Arguments:
            path (string): Path to directory with the dataset.
            extension (string): Extension of the files in the dataset.
            dataset_vars (dict): Dictionary with the variables needed to load the dataset.

            transform (callable, optional): Optional transform to be applied on a sample.
            preload (bool): If True, the dataset is preloaded in memory.
        """
        self.dataset_filenames = get_filenames(path, extension, **dataset_vars)
        self.data_reader = DATASET_READER[extension]
        self.dataset_vars = dataset_vars
        self.transform_input = transform_input
        self.transform_output = transform_output
        self.preload = preload or extension == 'builtin'

        if self.preload:
            self.dataset = [self.data_reader(f, **self.dataset_vars) for f in self.dataset_filenames]
            self.dataset = self.dataset[0] if extension == 'builtin' else dataset

    def __len__(self):
        return len(self.dataset_filenames)

    def __getitem__(self, idx):
        if self.preload:
            data, label = self.dataset[idx]
        else:
            data, label = self.data_reader(self.dataset_filenames[idx], **self.dataset_vars)

        if self.transform_input:
            data = self.transform_input(data)

        if self.transform_output:
            label = self.transform_output(label)

        return data, label


class Dataset:
    def __init__(self,
                 path,
                 extension,
                 dataset_vars=None,
                 transform_input=None,
                 transform_output=None,
                 preload=False,
                 use_loader=True,
                 batch_size=32,
                 shuffle=False,
                 num_workers=0):
        dataset = CustomDataset(path, extension, transform_input=transform_input, transform_output=transform_output,
                                dataset_vars=dataset_vars, preload=preload)

        if use_loader:
            self.train_loader = data.DataLoader(dataset, batch_size=batch_size,
                                                shuffle=shuffle, num_workers=num_workers)


if __name__ == '__main__':
    dataset_vars = dict(name='cifar10')
    dataset = Dataset('/home/enmartz/Downloads',
                      'builtin',
                      dataset_vars=dataset_vars,
                      transform_input=None,
                      transform_output=None,
                      preload=False,
                      use_loader=True,
                      batch_size=32,
                      shuffle=False,
                      num_workers=0)
    print(dataset.train_loader)
    for data, label in dataset.train_loader:
        print(data, label)
        break
