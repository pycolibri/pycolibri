
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np

from  .autoencoder import Autoencoder
from  .unet import Unet
import tensorflow as tf


class DIP_CASSI(tf.keras.models.Model):
    """
    DIP CASSI Model

    Args:
        input_shape (list): Shape of the input.
        recon (str): Reconstruction network.
        initilization (str): Type of initialization.
        optical_layer (tf layer): Optical layer.
        mode (str): CASSI mode.
        network_args (dict): Arguments for the reconstruction network.

    Returns:
        tf.keras.model: DIP Model

    """

    def __init__(self, input_shape=[128, 128, 25], recon='unet', initilization='random', optical_layer=None, mode='base', network_args=None):
        """
        Initializes the DIP CASSI Model.

        Args:
            input_shape (list): Shape of the input.
            recon (str): Reconstruction network.
            initilization (str): Type of initialization.
            optical_layer (tf layer): Optical layer.
            mode (str): CASSI mode.
            network_args (dict): Arguments for the reconstruction network.

        Returns:
            tf.keras.model: DIP Model

        """
        super(DIP_CASSI, self).__init__()
        self.initilization = initilization
        self.optics = optical_layer
        self.input_size = input_shape

        if recon == 'unet':
            self.recon = Unet(out_channels=input_shape[-1], features=network_args['features'], last_activation='relu')
        elif recon == 'autoencoder':
            self.recon = Autoencoder(out_channels=input_shape[-1], features=network_args['features'], last_activation='relu', reduce_spatial=network_args['reduce_spatial'])
        else:
            raise ValueError("Choose models autoencoder or unet")

    def __call__(self, y, training=True):
        """
        Perform forward pass.

        Args:
            y: Input data.
            training (bool): Training mode.

        Returns:
            ys: Forward pass result if training=True, else the reconstructed data (x).

        """
        if self.initilization == 'random':
            z0 = tf.random.normal(shape=(1, *self.input_size))
        if self.initilization == 'transpose':
            z0 = self.optics(y, type_calculation='backward')

        x = self.recon(z0)

        ys = self.optics(x, type_calculation='forward')

        if training:
            return ys
        else:
            return x
