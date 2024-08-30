.. _datasets:

Datasets
========

The PyColibri datasets module offers a variety of datasets widely used in machine learning and computer vision research, including both built-in datasets and custom datasets. This module simplifies the process of loading and transforming datasets, providing a consistent interface for researchers and developers to work with different data types.

Built-in Datasets
-----------------

The built-in datasets supported by this module include popular datasets like MNIST, CIFAR-10, CIFAR-100, and Fashion MNIST. The complete list is aviailable in the `colibri.data.utils.BUILTIN_DATASETS` dictionary.

.. autosummary::
    :toctree: stubs
    :template: class_template.rst
    :nosignatures:

    colibri.data.utils.load_builtin_dataset
    colibri.data.utils.update_builtin_path


Custom Dataset Class
--------------------

A flexible dataset class is available to handle both built-in and custom datasets. It includes methods for loading datasets and applying necessary transformations.

.. autosummary::
    :toctree: stubs
    :template: class_template.rst
    :nosignatures:

    colibri.data.datasets.CustomDataset

Custom Datasets
-----------------------

Custom datasets are supported by the Handlers, Currently, the module supports the following datasets: Cave

.. autosummary::
    :toctree: stubs
    :template: class_template.rst
    :nosignatures:

    colibri.data.sota_datasets.CaveDatasetHandler

Default Transformations
-----------------------

The module provides a default transformation class to handle the standard preprocessing tasks required for the input and output data. This includes conversion to tensors, among other operations, ensuring that the data is in the right format for model training.

.. autosummary::
    :toctree: stubs
    :template: class_template.rst
    :nosignatures:

    colibri.data.datasets.DefaultTransform
