import os

import numpy as np
from PIL import Image

from deepoptix.datasets.utils import *


class FolderDataset(tf.data.Dataset):
    """
    FolderDataset

    :param data_path: path to the dataset
    :rtype: FolderDataset
    """

    def _generator(data_path):
        filenames = get_all_filenames(data_path.decode())

        for filename in filenames:
            image = Image.open(filename)

            if image.mode != 'RGB':
                image = image.convert('RGB')

            yield np.array(image) / 255.

    def __new__(cls, data_path):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.float32),
            args=(data_path,)
        )


class Dataset:
    """
    Dataset

    :param data_path: path to the dataset. If it is a string, it will load a basic dataset,
                      otherwise if it is a dictionary it will load a folder dataset
    :param batch_size: batch size
    :param buffer_size: buffer size for shuffling
    :param chache_dir: directory to cache the dataset
    :rtype: Dataset
    """

    def __init__(self, data_path, batch_size, buffer_size=3, chache_dir=''):
        self.data_path = data_path
        self.buffer_size = buffer_size
        self.chache_dir = chache_dir

        basic_datasets = ['mnist, fashion_mnist', 'cifar10', 'cifar100']

        # basic datasets

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
        Function that loads basic datasets with labels or classes
        :param name: name of the dataset (mnist, fashion_mnist, cifar10, cifar100)
        :param batch_size: batch size
        :return: train and test dataset
        """
        print('Loading dataset: ', name)

        if name == 'mnist':
            dataset = tf.keras.datasets.mnist
        elif name == 'fashion_mnist':
            dataset = tf.keras.datasets.fashion_mnist
        elif name == 'cifar10':
            dataset = tf.keras.datasets.cifar10
        elif name == 'cifar100':
            dataset = tf.keras.datasets.cifar100
        else:
            raise ValueError('Dataset not supported')

        (x_train, y_train), (x_test, y_test) = dataset.load_data()

        # create dataset

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train.squeeze()))
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test.squeeze()))

        return train_dataset, test_dataset

    def build_pipeline(self, dataset, batch_size, shuffle=False, cache_dir=''):
        """
        Function that builds the pipeline for the dataset
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
