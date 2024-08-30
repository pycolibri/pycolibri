from setuptools import setup

setup(
    name='colibri',
    install_requires=['torch', 'torchmetrics', 'tqdm', 'torchvision', 'matplotlib', 'h5py', 'torch_dct'],
    extras_require={'doc': ['sphinx', 'sphinx_rtd_theme', 'autodocsumm', 'sphinx_gallery'],
                    'test': ['pytest']}
)
