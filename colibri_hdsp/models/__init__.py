from colibri_hdsp.models.custom_layers import *

from colibri_hdsp.models.unet import Unet
from  colibri_hdsp.models.autoencoder import Autoencoder
from colibri_hdsp.models.DIP import DIP_CASSI
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


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
    
if __name__ == "__main__":
    import tensorflow as tf


    model_layer = Autoencoder

    network_params = dict(
        out_channels=1,
        features=[32, 64, 128, 256],
        last_activation='relu',
        reduce_spatial = True
    )

    model = build_network(model=model_layer, size=32, in_channels=1, **network_params)
    model.summary()

    x = tf.random.normal((1, 32, 32, 1))
    y = model(x)
    
    print("input shape:", x.shape)
    print("output shape:", y.shape)






