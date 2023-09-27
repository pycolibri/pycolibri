from .unet import Unet
from .autoencoder import Autoencoder
from .DIP import DIP_CASSI
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

__all__ = [ 
    "Unet",
    "Autoencoder",
    "DIP_CASSI",
    "build_network"
]

def build_network(model=Unet, size=256, in_channels=31 ,**network_params):
    """ Build the network model

    Args:
        model_name (tf.keras.Layer, optional): Model name. Defaults to "unet".
        in_channels (int, optional): Number of input channels. Defaults to 1.
        size (int, optional): Size of the input image. Defaults to None.
        **network_params (dict): Network parameters
    """
    _input = Input(shape=(size, size, in_channels))
    output = model(**network_params)(_input)
    model = Model(inputs=_input, outputs=output)
    return model





