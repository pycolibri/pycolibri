from setuptools import setup

setup(
    name='colibri',
    install_requires=['torch>=1.13.0', 
                      'torchmetrics>=0.11.4', 
                      'tqdm>=4.66.5', 
                      'torchvision>=0.14.0', 
                      'matplotlib>=3.9.2', 
                      'h5py>=3.12.1', 
                      'torch_dct>=0.1.6', 
                      'requests>=2.32.3',
                      'scikit-image>=0.21.0'],
    extras_require={'doc': ['sphinx', 'sphinx_rtd_theme', 'autodocsumm', 'sphinx_gallery'],
                    'test': ['pytest']}
)
