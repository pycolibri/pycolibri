r"""
Demo Datasets.
===================================================

In this example we show how to use the custom dataset class to load the predefined datasets in the repository.

"""
from torch.utils import data

# %%
# Load dataset
# -----------------------------------------------
from colibri.data.datasets import CustomDataset

name = 'cifar10'
path = '.'
batch_size = 128

builtin_dict = dict(train=True, download=True)
dataset = CustomDataset(name, path,
                        builtin_dict=builtin_dict,
                        transform_dict=None)

dataset_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# %%
# Visualize dataset
# -----------------------------------------------

import matplotlib.pyplot as plt

data = next(iter(dataset_loader))
image = data['input']
label = data['output']

plt.figure(figsize=(5, 5))
plt.suptitle(f'{name.upper()} dataset Samples')

for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(image[i].permute(1, 2, 0).cpu().numpy())
    plt.axis('off')

plt.tight_layout()
plt.show()
