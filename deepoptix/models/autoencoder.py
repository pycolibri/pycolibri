""" Autoencoder Architecture """

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from layers import *
# TODO: Implement Autoencoder model

class Autoencoder(Model):
    """
    Autoencoder model


    :param out_channels: number of output channels
    :param levels: number of levels in the Autoencoder
    :param activation: activation function
    :param filter_list: list of filters
    :param reduce_spatial: select if the autoencoder reduce spatial dimension
    :param latent_channels: number of latent space channels
    :return: Autoencoder model
    :rtype: tf.keras.Model
    """

    def __init__(self,reduce_spatial=False, latent_channels = 3, levels=4, output_channel=1,kernel_size=3,activation='relu',filter_list=None):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(reduce_spatial=reduce_spatial, latent_channels = latent_channels, levels=levels, filter_list=filter_list[::-1],kernel_size=kernel_size,activation=activation)
        self.decoder = Decoder(reduce_spatial=reduce_spatial, output_channel=output_channel, levels=levels,activation=activation,
                               kernel_size=kernel_size, filter_list=filter_list)
    def call(self, inputs, get_latent=False,**kwargs):
        z = self.encoder(inputs)
        x = self.decoder(z)
        if get_latent:
            return x,z
        else:
            return x

class Encoder(Model):
    """
    Encoder model


    :param levels: number of levels in the autoencoder
    :param kernel_size: kernel size of the conv layers
    :param activation: activation function
    :param filter_list: list of filters
    :param reduce_spatial: select if the autoencoder reduce spatial dimension
    :param latent_channels: number of latent space channels
    :return: Encoder model
    :rtype: tf.keras.Model
    """
    def __init__(self, reduce_spatial=False, latent_channels = 3, levels=4, filter_list=None, kernel_size=3, activation='relu'):
        super(Encoder, self).__init__()

        self.conv_lay = []
        self.reduce_spatial = reduce_spatial
        self.levels = levels
        self.bn = []
        self.act = tf.keras.layers.Activation(activation)
        for i in range(levels):
            self.conv_lay.append(Conv2D(filter_list[i],kernel_size=kernel_size,padding='SAME'))
            self.bn.append(BatchNormalization())
        self.conv_lay.append(Conv2D(latent_channels, kernel_size=kernel_size,  padding='SAME'))
        if reduce_spatial:
            self.maxpool = MaxPooling2D(pool_size=(2,2))

    def call(self, input, **kwargs):
        x = input
        for i in range(self.levels):
            x = self.conv_lay[i](x)
            x = self.bn[i](x)
            x = self.act(x)
            if self.reduce_spatial:
                x = self.maxpool(x)
        x = self.conv_lay[-1](x)
        return x


class Decoder(Model):
    """
    Decoder model


    :param levels: number of levels in the autoencoder
    :param kernel_size: kernel size of the conv layers
    :param activation: activation function
    :param filter_list: list of filters
    :param reduce_spatial: select if the autoencoder reduce spatial dimension
    :param out_channels: number of latent space channels
    :return: Decoder model
    :rtype: tf.keras.Model
    """
    def __init__(self, reduce_spatial=False, levels=4, filter_list=None, kernel_size=3,output_channel=1,activation='relu'):
        super(Decoder, self).__init__()
        self.conv_lay = []
        self.reduce_spatial = reduce_spatial
        self.levels = levels
        self.bn = []
        self.act = tf.keras.layers.Activation(activation)
        for i in range(levels):
            self.conv_lay.append(Conv2D(filter_list[i],kernel_size=kernel_size, padding='SAME'))
            self.bn.append(BatchNormalization())
        self.conv_lay.append(Conv2D(output_channel, kernel_size=kernel_size, padding='SAME'))
        if reduce_spatial:
            self.ups = UpSampling2D(size=(2,2))

    def call(self, input, **kwargs):
        x = input
        for i in range(self.levels):
            x = self.conv_lay[i](x)
            x = self.bn[i](x)
            x = self.act(x)
            if self.reduce_spatial:
                x = self.ups(x)
        x = self.conv_lay[-1](x)
        return x



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import scipy.io as sio
    import os

    # load a mat file

    cube = sio.loadmat('spectral_image.mat')['img']


    # load optical encoder
    cube_tf = tf.convert_to_tensor(cube)[None]  # None add a new dimension
    print(cube_tf.shape)
    model = Autoencoder(reduce_spatial=True,latent_channels=1,filter_list=[4,16,32],output_channel = cube.shape[-1],kernel_size=3,levels=3)
    model.build(cube_tf.shape)  # this is only for the demo

    # encode the cube

    recon = model(cube_tf,True)


    # Print information about tensors


    # visualize the measurement

    plt.figure(figsize=(10, 10))

    plt.subplot(131)
    plt.title('cube')
    plt.imshow(cube[..., 0])

    plt.subplot(132)
    plt.title('latent space')
    plt.imshow(recon[1][0, ...])

    plt.subplot(133)
    plt.title('recon')
    plt.imshow(recon[0][0, ..., 0])

    plt.show()
