import matplotlib.pyplot as plt

from torch.utils import data
from torch.utils.data import Dataset

import colibri.data.utils as D

DATASET_READER = {
    'builtin': D.load_builtin,
    'img': D.load_img,
    'mat': D.load_mat,
    'h5': D.load_h5,
}


class CustomDataset(Dataset):
    """Custom dataset."""

    def __init__(self, path, extension, dataset_vars=None, transform_input=None, transform_output=None, preload=False):
        """
        Arguments:
            path (string): Path to directory with the dataset.
            extension (string): Extension of the files in the dataset.
            dataset_vars (dict): Dictionary with the variables needed to load the dataset.
            transform_input (callable, optional): Optional transform to be applied to the input data.
            transform_output (callable, optional): Optional transform to be applied to the output data.
            preload (bool): If True, the dataset will be loaded in memory.
        """
        self.dataset_filenames = D.get_filenames(path, extension, **dataset_vars)
        self.data_reader = DATASET_READER[extension]
        self.dataset_vars = dataset_vars
        self.transform_input = transform_input
        self.transform_output = transform_output
        self.preload = preload or extension == 'builtin'
        self.default_transform = D.DefaultTransform(extension)
        self.len_dataset = len(self.dataset_filenames)

        if self.preload:
            self.dataset = [self.data_reader(f, **self.dataset_vars) for f in self.dataset_filenames]
            self.dataset = self.dataset[0] if extension == 'builtin' else self.dataset
            self.len_dataset = len(self.dataset)

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        if self.preload:
            data, label = self.dataset[idx]
        else:
            data, label = self.data_reader(self.dataset_filenames[idx], **self.dataset_vars)

        if self.transform_input:
            data = self.transform_input(data)
        else:
            data = self.default_transform.transform_data(data)

        if self.transform_output:
            label = self.transform_output(label)
        else:
            label = self.default_transform.transform_label(label)

        return data, label


if __name__ == '__main__':
    dataset_vars = dict(name='cifar10', train=True, download=True)
    dataset = CustomDataset('/home/enmartz/Downloads',
                            'builtin',
                            dataset_vars=dataset_vars,
                            transform_input=None,
                            transform_output=None,
                            preload=False)

    dataset_loader = data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    # plot 3 x 3 images

    data, label = next(iter(dataset_loader))

    plt.figure(figsize=(5, 5))
    plt.suptitle('CIFAR10 dataset Samples')

    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(data[i].permute(1, 2, 0).cpu().numpy())
        plt.title(label[i].cpu().numpy())
        plt.axis('off')

    plt.tight_layout()
    plt.show()
