r"""
Demo Datasets.
===================================================

In this example we show how to use the custom dataset class to load the predefined datasets in the repository.

"""

# %%
# Select Working Directory and Device
# -----------------------------------------------
import os
from random import randint

from torch.utils.data import DataLoader

os.chdir(os.path.dirname(os.getcwd()))
print("Current Working Directory ", os.getcwd())

# %%
# Load builtin dataset
# -----------------------------------------------
from colibri.data.datasets import CustomDataset


name = 'fashion_mnist'  # ['cifar10', 'cifar100', 'mnist', 'fashion_mnist', 'cave']
path = 'data'
batch_size = 128
builtin_train = True
builtin_download = True

dataset = CustomDataset(name, path, builtin_train, builtin_download)
dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# %%
# Visualize cifar10 dataset
# -----------------------------------------------

import matplotlib.pyplot as plt

data = next(iter(dataset_loader))
image = data['input']
label = data['output']

plt.figure(figsize=(5, 5))
plt.suptitle(f'{name.upper()} dataset Samples - range: [{image.min()}, {image.max()}]')

for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(image[i].permute(2, 1, 0).cpu().numpy())
    plt.title(f'Label: {label[i]}')
    plt.axis('off')

plt.tight_layout()
plt.show()


# %%
# Cave dataset
# -----------------------------------------------
# CAVE is a database of multispectral images that were used to emulate the GAP camera.
# The images are of a wide variety of real-world materials and objects.
# You can download the dataset from the following link: http://www.cs.columbia.edu/CAVE/databases/multispectral/
# Once downloaded, you must extract the files and place them in the 'data' folder.

import requests
import zipfile
import os

dataset = CustomDataset("cave", path)

dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# %%
# Visualize cave dataset
# -----------------------------------------------

def normalize(image):
    return (image - image.min()) / (image.max() - image.min())

data = next(iter(dataset_loader))
rgb_image = data['input']
spec_image = data['output']

[_, _, M, N] = spec_image.shape

plt.figure(figsize=(8, 8))
plt.suptitle(f'CAVE dataset Samples - spectral range: [{spec_image.min():.2f}, {spec_image.max():.2f}] - '
             f'RGB range: [{rgb_image.min():.2f}, {rgb_image.max():.2f}]')

for i in range(3):
    coord1 = [randint(0, M-1), randint(0, N-1)]
    coord2 = [randint(0, M-1), randint(0, N-1)]

    plt.subplot(3, 3, (3 * i) + 1)
    plt.imshow(normalize(rgb_image[i].permute(1, 2, 0).cpu().numpy()))
    plt.title('rgb')
    plt.axis('off')

    plt.subplot(3, 3, (3 * i) + 2)
    plt.imshow(normalize(spec_image[i, [18, 12, 8]].permute(1, 2, 0).cpu().numpy()))
    plt.scatter(coord1[1], coord1[0], s=120, edgecolors='black')
    plt.scatter(coord2[1], coord2[0], s=120, edgecolors='black')
    plt.title('spec bands [18, 12, 8]')
    plt.axis('off')

    plt.subplot(3, 3, (3 * i) + 3)
    plt.plot(normalize(spec_image[i, :, coord1[0], coord1[1]].cpu().numpy()), linewidth=2, label='p1')
    plt.plot(normalize(spec_image[i, :, coord2[0], coord1[1]].cpu().numpy()), linewidth=2, label='p2')
    plt.title('spec signatures')
    plt.xlabel('Wavelength [nm]')
    plt.grid()
    plt.legend()

plt.tight_layout()
plt.show()
