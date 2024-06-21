from torch.utils import data
from torch.utils.data import Dataset

import colibri.data.utils as D

DATASET_READER = {
    'builtin': D.load_builtin_dataset,
    'img': D.load_img_dataset,
    'mat': D.load_mat_dataset,
    'h5': D.load_h5_dataset,
}


def get_filenames(path):
    """Return a list with the filenames of the files in the given path."""
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


class CustomDataset(Dataset):
    """Custom dataset."""

    def __init__(self, path, extension, transform=None, dataset_vars=None, apply_transform_label=False, preload=False):
        """
        Arguments:
            path (string): Path to directory with the dataset.
            extension (string): Extension of the files in the dataset.
            dataset_vars (dict): Dictionary with the variables needed to load the dataset.

            transform (callable, optional): Optional transform to be applied on a sample.
            preload (bool): If True, the dataset is preloaded in memory.
        """
        self.dataset_filenames = get_filenames(path)
        self.data_reader = DATASET_READER[extension]
        self.transform = transform
        self.apply_transform_label = apply_transform_label
        self.preload = preload

        if preload:
            self.dataset = [self.data_reader(f) for f in self.dataset_filenames]

    def __len__(self):
        return len(self.dataset_filenames)

    def __getitem__(self, idx):
        if self.preload:
            data, label = self.dataset[idx]
        else:
            data, label = self.data_reader(self.dataset_filenames[idx])

        if self.transform:
            data = self.transform(data)
            label = self.transform(label) if self.apply_transform_label else label

        return data, label


class Dataset:
    def __init__(self,
                 path,
                 extension,
                 transform=None,
                 dataset_vars=None,
                 apply_transform_label=True,
                 preload=False,
                 use_loader=True,
                 batch_size=32,
                 shuffle=False,
                 num_workers=0):
        dataset = CustomDataset(path, extension, dataset_vars, transform=transform,
                                apply_transform_label=apply_transform_label, preload=preload)

        if use_loader:
            self.train_loader = data.DataLoader(dataset, batch_size=batch_size,
                                                shuffle=shuffle, num_workers=num_workers)
