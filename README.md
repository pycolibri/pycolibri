<div style="display:flex;">
  <img src="docs/source/figures/colibri-logo.svg" alt="Image 1" style="width:30%;margin-left:auto;marging-right:auto;">
  <img src="docs/source/figures/colibri-banner.svg" alt="Image 2" style="width:60%;margin-left:auto;marging-right:auto;">
</div>

[![Python 3.6](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)]()

Colibri is a PyTorch library in development for solving computational imaging tasks where optical systems and
state-of-the-art deep neural networks are implemented to be easily used or modified for new research ideas. The purpose
of Colobri is to boost the research-related areas where optics and networks are required and introduce new researchers
to state-of-the-art algorithms in a straightforward and friendly manner.

## 💿 Installation

1. Clone the repository:

```bash
https://github.com/pycolibri/pycolibri.git
```

2. Create a virtual environment with conda:

```bash
conda create -n colibri python=3.10
conda activate colibri
```

3. Install the requirements:

```bash
pip install -r requirements.txt
```

4. Enjoy! 😄

## 🚀 Quick Start

You can go to ``examples`` folder and run cells of the notebook ``demo_colibri.ipynb`` to see how the library works.

### 💡 Examples

#### Dataset Visualization

```python
import matplotlib.pyplot as plt
from colibri_hdsp.data.datasets import Dataset
from torchvision.utils import make_grid

# Load dataset

dataset_path = 'cifar10'
keys = ''
batch_size = 128

dataset = Dataset(dataset_path, keys, batch_size)
adquistion_name = 'cassi' #  ['spc', 'cassi']

# get samples

sample = next(iter(dataset.train_dataset))[0]
img = make_grid(sample[:32], nrow=8, padding=1, normalize=True, range=None, scale_each=False, pad_value=0)

# visualize samples

plt.figure(figsize=(10,10))
plt.imshow(img.permute(1, 2, 0))
plt.title('CIFAR10 dataset')
plt.axis('off')
plt.show()

```

## 🧰 Features

- Flexible and customizable code.
- Train end-to-end models in an straightforward manner.
- Easily modify the implemented models.
- Easily add new optical systems.
- Easily add new deep neural networks.

## Available Models

### 📷 Optical Systems

- Coded Aperture Snapshot Spectral Imager (CASSI).
- Single Pixel Snapshot Imager (SPSI).

### 🖥️ Deep Neural Networks

- Autoencoder.
- Unet.

## 🎆 Frameworks

- End-to-end framework with optical systems as encoder models and deep neural networks as decoder models.

## 💡 Contributing

We welcome contributions from the community! If you have something you'd like to share, please follow these steps:

1. **Fork** the repository.
2. **Add** your improvement.
3. **Submit** a Pull Request.

## 🛡️ License

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```