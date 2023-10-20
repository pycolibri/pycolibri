import h5py

import torch
import torchvision
from torch.utils import data
from torchvision import transforms

from spec2rgb import ColourSystem
from utils import *

BASIC_DATASETS = {
    'mnist': torchvision.datasets.MNIST,
    'fashion_mnist': torchvision.datasets.FashionMNIST,
    'cifar10': torchvision.datasets.CIFAR10,
    'cifar100': torchvision.datasets.CIFAR100,
}


class FolderDataset(data.Dataset):
    def __init__(self, dataset_path, keys=None, is_train=False):
        super(FolderDataset, self).__init__()
        self.keys = keys

        self.filenames = get_all_filenames(dataset_path)
        self.dataset_len = len(self.filenames)

        self.is_train = is_train

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):

        # load sample

        filename = self.filenames[index]

        # from jpg, jpeg or png images

        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            sample = load_image(filename)

            return torch.from_numpy(sample.copy()).permute(2, 0, 1)

        # from mat files

        elif filename.lower().endswith('.mat'):
            with h5py.File(str(filename), 'r') as mat:
                spectral_image = np.float32(np.array(mat[self.keys['spec']]))
                spectral_image = np.transpose(spectral_image, (2, 1, 0))

            filename = filename.replace('_spectral', '_RGB')
            image = load_image(filename.replace('.mat', '.jpg'))

            spectral_image = torch.from_numpy(spectral_image.copy()).permute(2, 0, 1)
            image = torch.from_numpy(image.copy()).permute(2, 0, 1)

            return image, spectral_image

        else:
            raise "Unknown filename extension (only availabe .jpg, .jpeg, .png and .mat files)"


class Dataset:
    """
    A class for managing and loading data.
    """

    def __init__(self, dataset_path, keys=None, batch_size=1, num_workers=0):
        """
        Initialize the Dataset class.

        Args:
            dataset_path (str or dict): Path to the dataset. If it is a string, it will load a basic dataset,
                                     otherwise, if it is a dictionary, it will load a folder dataset.
            keys (str): If the dataset contains .mat files, it should contain the keys related to each
                         sample. For arad, we have keys=dict(spec='cube')
            batch_size (int): Batch size.
            num_workers (int): number of workers.
        """
        self.dataset_path = dataset_path

        # basic data

        basic_datasets = list(BASIC_DATASETS.keys())
        if isinstance(dataset_path, str):
            cmp_datasets = [dataset_name in dataset_path for dataset_name in basic_datasets]

            if any(cmp_datasets):
                idx = np.argwhere(cmp_datasets)[0][0]
                train_dataset, test_dataset = self.load_basic_dataset(basic_datasets[idx])

            else:
                raise ValueError('Dataset not supported')

        # folder data

        elif isinstance(dataset_path, dict):
            train_data_path = dataset_path['train']
            test_data_path = dataset_path['test']

            train_dataset = FolderDataset(train_data_path, keys=keys)
            test_dataset = FolderDataset(test_data_path, keys=keys)

        else:
            raise ValueError('Dataset not supported')

        self.train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                         num_workers=num_workers)
        self.test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                        num_workers=num_workers)

    def load_basic_dataset(self, name):
        """
        Load basic data with labels or classes.

        Args:
            name (str): Name of the dataset (mnist, fashion_mnist, cifar10, cifar100).

        Returns:
            tuple: Train and test data.
        """
        print('Loading dataset: ', name)

        try:
            dataset = BASIC_DATASETS[name]
        except:
            raise ValueError('Dataset not supported')

        transform = transforms.Compose([transforms.ToTensor()])

        train_dataset = dataset(root='./data', train=True, download=True, transform=transform)
        test_dataset = dataset(root='./data', train=False, download=True, transform=transform)

        return train_dataset, test_dataset


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # load dataset
    dataset_path = 'cifar10'
    keys = ''
    # dataset_path = dict(train='/home/myusername/Datasets/processed/coco/train2017_r512',
    #                  test='/home/myusername/Datasets/processed/coco/train2017_r512')
    # dataset_path = dict(train='/home/myusername/Datasets/raw/arad/Train_spectral',
    #                  test='/home/myusername/Datasets/raw/arad/Valid_spectral')
    # keys = dict(spec='cube')
    batch_size = 32

    dataset = Dataset(dataset_path, keys=keys, batch_size=batch_size)
    train_dataset, test_dataset = dataset.train_dataset, dataset.test_dataset

    # print information about dataset
    print('train dataset: ', train_dataset)
    print('test dataset: ', test_dataset)

    # visualize samples

    if isinstance(dataset_path, str):  # basic datasets
        basic_datasets = list(BASIC_DATASETS.keys())
        cmp_datasets = [dataset_name in dataset_path for dataset_name in basic_datasets]

        if any(cmp_datasets):
            # visualize dataset
            for x, y in train_dataset:
                plt.figure(figsize=(7, 7))
                for i in range(9):
                    plt.subplot(3, 3, i + 1)
                    plt.imshow(x[i].permute(1, 2, 0), cmap='gray')
                    plt.title(f'class number: {y[i]}')
                    plt.axis('off')

                plt.tight_layout()
                plt.show()

                break

    elif 'arad' in dataset_path['train']:  # arad dataset
        cs_ciergb = ColourSystem(start=400, end=700, num=31)
        RGB = cs_ciergb.get_transform_matrix()

        for x, y in train_dataset:
            plt.figure(figsize=(7, 7))
            plt.suptitle('rgb samples')
            for i in range(9):
                plt.subplot(3, 3, i + 1)
                plt.imshow(x[i].permute(1, 2, 0))
                plt.axis('off')

            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(7, 7))
            plt.suptitle('mapped spectral samples')
            for i in range(9):
                rgb = cs_ciergb.spec_to_rgb(y[i].permute(1, 2, 0).numpy())
                plt.subplot(3, 3, i + 1)
                plt.imshow(rgb)
                plt.axis('off')

            plt.tight_layout()
            plt.show()

            break

    else:  # folder dataset
        for x in train_dataset:
            plt.figure(figsize=(7, 7))
            for i in range(9):
                plt.subplot(3, 3, i + 1)
                plt.imshow(x[i].permute(1, 2, 0))
                plt.axis('off')

            plt.tight_layout()
            plt.show()

            break
