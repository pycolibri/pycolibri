<div style="display:flex;">
  <img src="docs/source/figures/colibri-banner.svg" alt="colibri-banner-full" style="width:100%;margin-left:auto;marging-right:auto;">
</div>

[![Python 3.6](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pycolibri/pycolibri/blob/main/main.ipynb)

``Colibri`` is a PyTorch library in development for solving computational imaging tasks where optical systems and
state-of-the-art deep neural networks are implemented to be easily used or modified for new research ideas. The purpose
of ``Colibri`` is to boost the research-related areas where optics and networks are required and introduce new researchers
to state-of-the-art algorithms in a straightforward and friendly manner.

## 🧰 Features

* A [datasets module](https://pycolibri.github.io/pycolibri/datasets.html) with common datasets for computational imaging tasks.
* A [collection of advanced architectures](https://pycolibri.github.io/pycolibri/architectures.html) that integrates optical systems with deep learning models.
* A module of [deep learning models](https://pycolibri.github.io/pycolibri/models.html)  for computational imaging tasks.
* A [collection of optical systems](https://pycolibri.github.io/pycolibri/optics.html) for spectral, depth, phase and others imaging applications.
* A set of [regularization functions](https://pycolibri.github.io/pycolibri/regularizers.html) to force physical constraints on the optical coding elements.
* A [recovery module](https://pycolibri.github.io/pycolibri/recovery.html) with state-of-the-art recovery algorithms used in image restoration on inverse problems.
* A set of well-explained [examples](https://pycolibri.github.io/pycolibri/auto_examples/index.html) demonstrating how to use the features of``Colibri``.


## Available Models

### 📷 [Optical Systems](https://pycolibri.github.io/pycolibri/optics.html#)

- Coded Aperture Snapshot Spectral Imager (CASSI).
    - Single Disperser CASSI [(SD-CASSI)](https://pycolibri.github.io/pycolibri/stubs/colibri.optics.cassi.SD_CASSI.html#colibri.optics.cassi.SD_CASSI)
    - Dual Disperser CASSI [(DD-CASSI)](https://pycolibri.github.io/pycolibri/stubs/colibri.optics.cassi.DD_CASSI.html)
    - Color CASSI [(C-CASSI)](https://pycolibri.github.io/pycolibri/stubs/colibri.optics.cassi.C_CASSI.html)
- Single Pixel Camera [(SPC)](https://pycolibri.github.io/pycolibri/stubs/colibri.optics.spc.SPC.html)
- Single Diffractive Optical Element [(DOE)](https://pycolibri.github.io/pycolibri/stubs/colibri.optics.doe.SingleDOESpectral.html) for Spectral Imaging. 

### 💻️ [Deep Neural Networks](https://pycolibri.github.io/pycolibri/models.html)

- [Autoencoder](https://pycolibri.github.io/pycolibri/models.html)
- [Unet](https://pycolibri.github.io/pycolibri/models.html)

### 🖥 [Recovery Algorithms](https://pycolibri.github.io/pycolibri/recovery.html)

- Fast Iterative Shrinkage-Thresholding Algorithm [(FISTA)](https://pycolibri.github.io/pycolibri/recovery.html)
- Plug-and-Play algorithm with Alternating Direction Method of Multipliers [(Plug-and-Play ADMM)](https://pycolibri.github.io/pycolibri/stubs/colibri.recovery.pnp.PnP_ADMM.html)

## 🎆 [Frameworks](https://pycolibri.github.io/pycolibri/architectures.html)

- [End-to-end framework](https://pycolibri.github.io/pycolibri/stubs/colibri.misc.e2e.E2E.html#colibri.misc.e2e.E2E) with optical systems as encoder models and deep neural networks as decoder models.


## 📑 [Documentation](https://pycolibri.github.io/pycolibri/index.html)

The documentation is available at [pycolibri.github.io/pycolibri](https://pycolibri.github.io/pycolibri/).

## 💿 Installation

1. Clone the repository:

```bash
git clone https://github.com/pycolibri/pycolibri.git
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
from torch.utils.data import DataLoader
from colibri.data.datasets import CustomDataset

# Load dataset

name = 'cifar10'  # ['cifar10', 'cifar100', 'mnist', 'fashion_mnist', 'cave']
path = 'data'
batch_size = 128

builtin_dict = dict(train=True, download=True)
dataset = CustomDataset(name, path,
                        builtin_dict=builtin_dict,
                        transform_dict=None)

dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Visualize dataset

import matplotlib.pyplot as plt

data = next(iter(dataset_loader))
image = data['input']
label = data['output']

plt.figure(figsize=(5, 5))
plt.suptitle(f'{name.upper()} dataset Samples')

for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(image[i].permute(1, 2, 0).cpu().numpy())
    plt.title(f'Label: {label[i]}')
    plt.axis('off')

plt.tight_layout()
plt.show()

```




## 💡 Contributing

Contributions are welcome! If you're interested in improving Colibri, please:

1. Fork the repository.
2. Create your feature branch (``git checkout -b feature/AmazingFeature``).
3. Commit your changes (``git commit -am 'Add some AmazingFeature'``).
4. Push to the branch (``git push origin feature/AmazingFeature``).
5. Run [build_and_test.sh](https://github.com/pycolibri/pycolibri/blob/150-readme-%2B-index/quick_validation.sh) to clear out old documentation, rebuild the documentation, run tests, and then open the newly generated documentation in a web browser.

See our [contribution guide](https://pycolibri.github.io/pycolibri/contributing.html) for more details.

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
