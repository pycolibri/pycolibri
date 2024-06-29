import matplotlib.pyplot as plt

from torch.utils import data
from torch.utils.data import Dataset

import colibri.data.utils as D

DATASET_READER = {
    'builtin': D.load_builtin_dataset,
    'img': D.load_img,
    'mat': D.load_mat,
    'h5': D.load_h5,
}


class CustomDataset(Dataset):
    """Custom dataset."""

    def __init__(self, path, extension, data_dict, data_name_dict=None, transform_dict=None, preload=False):
        """
        Arguments:
            path (string): Path to directory with the dataset.
            extension (string): Extension of the dataset.
            data_dict (dict): Dictionary with the variables needed to load the dataset.
            data_name_dict (dict): Dictionary with the names of the data.
            transform_dict (dict): Dictionary with the transformations to apply to the data.
            preload (bool): If True, the dataset will be loaded in memory.
        """
        assert 'builtin' in extension or data_name_dict is not None, ('data_name_dict must be provided '
                                                                      'for non-builtin datasets')
        if data_name_dict is not None:
            assert 'input' in data_name_dict, 'input key must be provided in data_name_dict'

        self.dataset_filenames = D.get_filenames(path, extension, **data_dict)
        self.data_reader = DATASET_READER[extension]
        self.data_dict = data_dict
        self.data_name_dict = data_name_dict
        self.transform_dict = transform_dict
        self.preload = preload or extension == 'builtin'
        self.default_transform = D.DefaultTransform(extension)
        self.len_dataset = len(self.dataset_filenames)

        if self.preload:
            if extension == 'builtin':
                builtin_dataset = self.data_reader(self.dataset_filenames[0], **self.data_dict)
                self.dataset = dict(input=builtin_dataset.data, output=builtin_dataset.targets)

            else:
                self.dataset = {}
                for key, data_name in self.data_name_dict.items():
                    name = D.get_name_from_key(key)
                    self.dataset[name] = [self.data_reader(f, data_name, **self.data_dict) for f in
                                          self.dataset_filenames]

            self.len_dataset = len(self.dataset['input'])

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):

        # load sample

        if self.preload:
            data = {key: value[idx] for key, value in self.dataset.items()}
        else:
            data = self.data_reader(self.dataset_filenames[idx], **self.data_dict)

        # apply transformation

        for key, transform in self.transform_dict.items():
            if key in data:
                data[key] = transform(data[key])

        return data


if __name__ == '__main__':
    data_dict = dict(name='mnist', train=True, download=True)
    data_name_dict = dict(input='data', output='label')
    dataset = CustomDataset('/home/enmartz/Downloads',
                            'builtin',
                            data_dict=data_dict,
                            transform_dict={},
                            preload=False)

    dataset_loader = data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    # plot 3 x 3 images

    data = next(iter(dataset_loader))
    image = data['input']
    label = data['output']

    plt.figure(figsize=(5, 5))
    plt.suptitle('CIFAR10 dataset Samples')

    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(image[i].permute(1, 2, 0).cpu().numpy())
        plt.title(label[i].cpu().numpy())
        plt.axis('off')

    plt.tight_layout()
    plt.show()
