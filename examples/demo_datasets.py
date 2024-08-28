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



dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


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

from colibri.data.sota_datasets import CaveDatasetHandler

cave_handler = CaveDatasetHandler(path='data', download=True)

dataset = CustomDataset(name, path,
                        builtin_dict=None,
                        transform_dict=None)

dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# %%
# Visualize cave dataset
# -----------------------------------------------

data = next(iter(dataset_loader))
rgb_image = data['input']
spec_image = data['output']

[_, _, M, N] = spec_image.shape

plt.figure(figsize=(8, 8))

for i in range(3):
    coord1 = [randint(0, M), randint(0, N)]
    coord2 = [randint(0, M), randint(0, N)]

    plt.subplot(3, 3, (3 * i) + 1)
    plt.imshow(rgb_image[i].permute(1, 2, 0).cpu().numpy())
    plt.title('rgb')
    plt.axis('off')

    plt.subplot(3, 3, (3 * i) + 2)
    plt.imshow(spec_image[i, [18, 12, 8]].permute(1, 2, 0).cpu().numpy())
    plt.scatter(coord1[1], coord1[0], s=120, edgecolors='black')
    plt.scatter(coord2[1], coord2[0], s=120, edgecolors='black')
    plt.title('spec bands [18, 12, 8]')
    plt.axis('off')

    plt.subplot(3, 3, (3 * i) + 3)
    plt.plot(spec_image[i, :, coord1[0], coord1[1]].cpu().numpy(), linewidth=2, label='p1')
    plt.plot(spec_image[i, :, coord2[0], coord1[1]].cpu().numpy(), linewidth=2, label='p2')
    plt.title('spec signatures')
    plt.xlabel('Wavelength [nm]')
    plt.grid()
    plt.legend()

plt.tight_layout()
plt.show()
