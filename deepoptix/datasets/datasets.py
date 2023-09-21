import os

import numpy as np
from PIL import Image

from deepoptix.datasets.utils import *

BASIC_DATASETS = {
    'mnist': tf.keras.datasets.mnist,
    'fashion_mnist': tf.keras.datasets.fashion_mnist,
    'cifar10': tf.keras.datasets.cifar10,
    'cifar100': tf.keras.datasets.cifar100
}


class FolderDataset(tf.data.Dataset):
    """
    Class for loading a dataset from a folder
    Args:
        data_path (str): path to the dataset
    """

    def _generator(data_path):
        """
        A generator function to load and preprocess images from the specified folder.

        Args:
            data_path (str): The path to the dataset folder.

        Yields:
            np.ndarray: Preprocessed image data as NumPy arrays, normalized to the range [0, 1].
        """
        filenames = get_all_filenames(data_path.decode())

        for filename in filenames:
            image = Image.open(filename)

            if image.mode != 'RGB':
                image = image.convert('RGB')

            yield np.array(image) / 255.

    def __new__(cls, data_path):
        """
        Create a new FolderDataset instance.

        Args:
            data_path (str): The path to the dataset folder.

        Returns:
            tf.data.Dataset: A TensorFlow Dataset containing preprocessed image data.
        """
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.float32),
            args=(data_path,)
        )


class Dataset:
    """
    A class for managing and loading datasets.

    Parameters:
        data_path (str or dict): Path to the dataset. If it is a string, it will load a basic dataset,
                                 otherwise, if it is a dictionary, it will load a folder dataset.
        batch_size (int): Batch size.
        buffer_size (int): Buffer size for shuffling (default is 3).
        cache_dir (str): Directory to cache the dataset.

    Returns:
        tuple: A tuple containing train and test datasets.
    """

    def __init__(self, data_path, batch_size, buffer_size=3, chache_dir=''):
        """
        Initialize the Dataset class.

        Args:
            data_path (str or dict): Path to the dataset. If it is a string, it will load a basic dataset,
                                     otherwise, if it is a dictionary, it will load a folder dataset.
            batch_size (int): Batch size.
            buffer_size (int, optional): Buffer size for shuffling (default is 3).
            cache_dir (str, optional): Directory to cache the dataset.
        """
        self.data_path = data_path
        self.buffer_size = buffer_size
        self.chache_dir = chache_dir

        # basic datasets

        basic_datasets = list(BASIC_DATASETS.keys())
        if isinstance(data_path, str):
            cmp_datasets = [dataset_name in data_path for dataset_name in basic_datasets]

            if any(cmp_datasets):
                idx = np.argwhere(cmp_datasets)[0][0]
                train_dataset, test_dataset = self.load_basic_dataset(basic_datasets[idx])

            else:
                raise ValueError('Dataset not supported')

        # folder datasets

        elif isinstance(data_path, dict):
            train_data_path = data_path['train']
            test_data_path = data_path['test']

            train_dataset = FolderDataset(train_data_path)
            test_dataset = FolderDataset(test_data_path)

        else:
            raise ValueError('Dataset not supported')

        self.train_dataset = self.build_pipeline(train_dataset, batch_size, shuffle=True, cache_dir=self.chache_dir)
        self.test_dataset = self.build_pipeline(test_dataset, batch_size, shuffle=False, cache_dir=self.chache_dir)

    def load_basic_dataset(self, name):
        """
        Load basic datasets with labels or classes.

        Args:
            name (str): Name of the dataset (mnist, fashion_mnist, cifar10, cifar100).

        Returns:
            tuple: Train and test datasets.
        """
        print('Loading dataset: ', name)

        try:
            dataset = BASIC_DATASETS[name]
        except:
            raise ValueError('Dataset not supported')

        (x_train, y_train), (x_test, y_test) = dataset.load_data()

        # create dataset

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train.squeeze()))
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test.squeeze()))

        return train_dataset, test_dataset

    def build_pipeline(self, dataset, batch_size, shuffle=False, cache_dir=''):
        """
        Build the data pipeline for the dataset.

        Args:
            dataset (tf.data.Dataset): The input dataset.
            batch_size (int): Batch size.
            shuffle (bool): Whether to shuffle the dataset.
            cache_dir (str): Directory to cache the dataset.

        Returns:
            tf.data.Dataset: The processed dataset.
        """
        dataset = dataset.cache(cache_dir) if cache_dir else dataset
        dataset = dataset.shuffle(self.buffer_size) if shuffle else dataset
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset


if __name__ == "__main__":

    import tensorflow as tf
    import matplotlib.pyplot as plt

    # load dataset
    data_path = 'cifar10'
    # data_path = dict(train='/home/myusername/Datasets/processed/coco/train2017_r512',
    #                  test='/home/myusername/Datasets/processed/coco/train2017_r512')
    batch_size = 32

    dataset = Dataset(data_path, batch_size)
    train_dataset, test_dataset = dataset.train_dataset, dataset.test_dataset

    # print information about dataset
    print('train dataset: ', train_dataset)
    print('test dataset: ', test_dataset)

    try:
        # visualize dataset
        for x, y in train_dataset.take(1):
            plt.figure(figsize=(7, 7))
            for i in range(9):
                plt.subplot(3, 3, i + 1)
                plt.imshow(x[i], cmap='gray')
                plt.title(f'class number: {y[i]}')
                plt.axis('off')

            plt.tight_layout()
            plt.show()

    except Exception as e:
        for x in train_dataset.take(1):
            plt.figure(figsize=(7, 7))
            for i in range(9):
                plt.subplot(3, 3, i + 1)
                plt.imshow(x[i])
                plt.axis('off')

            plt.tight_layout()
            plt.show()
