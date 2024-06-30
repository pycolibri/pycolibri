from setuptools import setup

setup(
    name='colibri',
    install_requires=['torch', 'torchmetrics', 'tqdm', 'torchvision', 'matplotlib', 'h5py', 'torch_dct'],
    extras_require={'doc': ['sphinx', 'furo', 'autodocsumm', 'sphinx_gallery'],
                    'test': ['pytest']}
)