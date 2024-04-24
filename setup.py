from setuptools import setup

setup(
    name='colibri',
    install_requires=['torch', 'torchmetrics', 'tqdm', 'torchvision', 'matplotlib', 'h5py'],
    extras_require={'doc': ['sphinx', 'furo', 'autodocsumm', 'sphinx_gallery', 'torch_dct']},
)