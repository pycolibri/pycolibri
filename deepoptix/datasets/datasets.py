import tensorflow as tf

from deepoptix.datasets.utils import *


def load_basic_dataset(name, batch_size):
    """
    Function that loads basic datasets with labels or classes
    :param name: name of the dataset (mnist, fashion_mnist, cifar10, cifar100)
    :param batch_size: batch size
    :return: train and test dataset
    """
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

    # dataset pipeline (fast lecture in some cases)

    train_dataset = train_dataset.map(lambda x, y: (normalize_img(x), hot_encode(y)))
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(batch_size)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    test_dataset = test_dataset.map(lambda x, y: (normalize_img(x), hot_encode(y)))
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.cache()
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset


if __name__ == "__main__":

    import tensorflow as tf
    import matplotlib.pyplot as plt

    # load dataset
    dataset_name = 'cifar10'
    batch_size = 32
    train_dataset, test_dataset = load_basic_dataset(dataset_name, batch_size)

    # print information about dataset
    print('train dataset: ', train_dataset)
    print('test dataset: ', test_dataset)

    # visualize dataset
    for x, y in train_dataset.take(1):
        print('x shape: ', x.shape)
        print('y shape: ', y.shape)
        plt.figure(figsize=(7, 7))
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(x[i], cmap='gray')
            plt.title(f'class number: {tf.argmax(y[i])}')
            plt.axis('off')

        plt.tight_layout()
        plt.show()
