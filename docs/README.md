
![colibri-banner](https://github.com/colibri-hdsp/colibri-hdsp/assets/58752635/ea337489-b8c5-473b-9431-2a3ca785a6de)

[![version](https://badge.fury.io/py/autodistill.svg?)](https://badge.fury.io/py/autodistill)
[![downloads](https://img.shields.io/pypi/dm/autodistill)](https://pypistats.org/packages/autodistill)
[![license](https://img.shields.io/pypi/l/autodistill?)](https://github.com/autodistill/autodistill/blob/main/LICENSE)
[![python-version](https://img.shields.io/pypi/pyversions/autodistill)](https://badge.fury.io/py/autodistill)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-auto-train-yolov8-model-with-autodistill.ipynb)

Colibri is a PyTorch library in development for solving computational imaging tasks where optical systems and state-of-the-art deep neural networks are implemented to be easily used or modified for new research ideas. The purpose of Colobri is to boost the research-related areas where optics and networks are required and introduce new researchers to state-of-the-art algorithms in a straightforward and friendly manner.

## 💿 Installation

Nunc felis lacus, mattis sit amet sollicitudin commodo, pharetra at erat. Nam eget risus at risus laoreet mattis ut in lacus. Pellentesque ipsum nisl, molestie quis aliquet eu, vestibulum vel risus.

1. Pellentesque varius metus leo, et viverra libero mattis a.
2. Nulla sed felis a massa auctor vulputate molestie eu justo.
3. Integer porttitor, sapien sit amet aliquam blandit, risus urna bibendum purus, nec consectetur velit eros ac lorem.
4. Morbi suscipit tristique viverra. Nulla scelerisque sit amet odio eu pretium. Nullam sagittis lorem et arcu egestas vehicula.

Done! ✔️

## 🚀 Quick Start

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed a facilisis dolor. In tempus varius tortor. Vestibulum ullamcorper dapibus vestibulum. Mauris semper dapibus magna, vitae iaculis ante gravida sed. Phasellus auctor, eros at rhoncus euismod, est mi lobortis tortor, ac feugiat massa neque sit amet ex. Aliquam nisi mi, feugiat ut mauris pellentesque, facilisis malesuada tortor. Ut tempor, lacus vitae vulputate imperdiet, urna dui finibus erat, non imperdiet lacus nulla non mi. Pellentesque ipsum nulla, finibus quis lacinia quis, consequat molestie eros.

### 💡 Examples

```
def factorial(n):
    """
    Calculate the factorial of a non-negative integer n.

    Args:
        n (int): Non-negative integer

    Returns:
        int: Factorial of n
    """
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

# Example usage
number = 5
print(f"The factorial of {number} is {factorial(number)}")
```


## 🧰 Features

- Quisque lobortis ultricies vestibulum.
- Phasellus euismod placerat lectus, eget condimentum turpis tempor viverra.
- Integer id metus ante. Phasellus dictum at arcu nec commodo.
- Fusce ultrices semper nibh quis aliquet.
- Phasellus finibus accumsan ipsum, quis laoreet purus scelerisque gravida.
- Donec ac metus dapibus, maximus nunc ac, sagittis arcu.

## Available Models

### 📷 Optical Systems

Here we are going to introduce the implemented optical systems.

- Coded Aperture Snapshot Spectral Imager (CASSI).
- Single Pixel Snapshot Imager (SPSI).

### 🖥️ Deep Neural Networks

Here we are going to introduce the implemented deep neural networks.

- Autoencoder.
- Unet.

## 🎆 Frameworks

Here we are going to introduce the implemented frameworks (end-to-end framework).

- End-to-end framework.

## 💡 Contributing

We welcome contributions from the community! If you have something you'd like to share, please follow these steps:

1. **Fork** the repository
2. **Add** your improvement
3. **Submit** a Pull Request

## 🛡️ License

Etiam sagittis varius ex tempus bibendum. Nunc non aliquam velit, id gravida augue. Maecenas faucibus efficitur quam sed convallis. Donec aliquet metus ac ligula ultricies, eget hendrerit nisi pellentesque. Aliquam commodo ligula quis rhoncus luctus. Cras ex velit, posuere non eros id, cursus egestas magna. Fusce a augue erat.

---
# how to execute the sphinx documentation 

This documentation is built using [Sphinx](https://www.sphinx-doc.org/en/master/), a tool that makes it easy to create intelligent and beautiful documentation, for execute the documentation you need to follow the next steps. U need a terminal to execute the commands in this directory.

1. Install the requirements
```bash
pip install -r requirements.txt
```

2. Build the documentation
```bash
make html
```

3. Open the documentation
```bash
open _build/html/index.html
```

4. Clean the documentation
```bash
make clean
```
