from setuptools import setup

setup(
    name='colibri_hdsp',
    install_requires=['torch', 'torchmetrics', 'tqdm', 'torchvision'],
    extras_require={'doc': ['sphinx', 'furo', 'autodocsumm', 'sphinx_gallery']},
)