Models
======

The models module within our library provides implementations of deep learning models tailored for computational imaging :math:`\mathcal{G}_\theta`. These models leverage the latest advancements in neural networks to offer robust solutions for image reconstruction, enhancement, and segmentation.


List of models
--------------------
The models module contains the following models:



.. autosummary::
    :toctree: stubs
    :template: class_template.rst
    :nosignatures:

    colibri_hdsp.models.autoencoder.Autoencoder
    colibri_hdsp.models.unet.Unet
    

List of custom layers
-----------------------

The custom_layers module within our library provides implementations of custom layers that are used in the models. These layers are designed to enhance the performance of the models by providing additional flexibility and control over the network architecture.



.. autosummary::
    :toctree: stubs
    :template: class_template.rst
    :nosignatures:

    colibri_hdsp.models.custom_layers.Activation
    colibri_hdsp.models.custom_layers.convBlock
    colibri_hdsp.models.custom_layers.downBlock
    colibri_hdsp.models.custom_layers.upBlock
    colibri_hdsp.models.custom_layers.upBlockNoSkip
    colibri_hdsp.models.custom_layers.outBlock

