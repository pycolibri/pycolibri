""" Autoencoder Architecture """

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
# TODO: Implement Autoencoder model

class Autoencoder(Model):
    def __init__(self,reduce_spatial=False, l_ch = 3, num_layers=4, dim_last=1,num_filt=8,kernel_size=3,activation='relu'):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(reduce_spatial=reduce_spatial, l_ch = l_ch, num_layers=num_layers, num_filt=num_filt,kernel_size=kernel_size,activation=activation)
        self.decoder = Decoder(reduce_spatial=reduce_spatial, dim_last=dim_last, num_layers=num_layers, num_filt=num_filt,
                               kernel_size=kernel_size, activation=activation)
    def call(self, inputs, **kwargs):
        z = self.encoder(inputs)
        x = self.decoder(z)
        return x

class Encoder(Model):
    def __init__(self,reduce_spatial=False, l_ch = 3, num_layers=4, num_filt=8,kernel_size=3,activation='relu'):
        super(Encoder, self).__init__()
        self.conv_lay = []
        self.reduce_spatial = reduce_spatial
        self.num_layers = num_layers
        for i in range(num_layers-1):
            self.conv_lay.append(Conv2D(num_filt*(i+1),kernel_size=kernel_size,activation=activation,padding='SAME'))
        self.conv_lay.append(Conv2D(l_ch, kernel_size=kernel_size, activation=activation, padding='SAME'))
        if reduce_spatial:
            self.maxpool = MaxPooling2D(pool_size=(2,2))

    def call(self, input, **kwargs):
        x = input
        for i in range(self.num_layers):
            x = self.conv_lay[i](x)
            if self.reduce_spatial:
                x = self.maxpool(x)

        return x


class Decoder(Model):
    def __init__(self, reduce_spatial=False, dim_last = 1, num_layers=4, num_filt=8, kernel_size=3, activation='relu'):
        super(Decoder, self).__init__()
        self.conv_lay = []
        self.reduce_spatial = reduce_spatial
        self.num_layers = num_layers
        for i in range(num_layers - 1):
            self.conv_lay.append(
                Conv2D(num_filt * (i + 1), kernel_size=kernel_size, activation=activation, padding='SAME'))
        self.conv_lay.append(Conv2D(dim_last, kernel_size=kernel_size, activation=activation, padding='SAME'))
        if reduce_spatial:
            self.ups = UpSampling2D(size=(2, 2))

    def call(self, input, **kwargs):
        x = input
        for i in range(self.num_layers):
            x = self.conv_lay[i](x)
            if self.reduce_spatial:
                x = self.ups(x)
        return x

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import scipy.io as sio
    import os

    # load a mat file

    # cube = sio.loadmat(os.path.join('examples', 'data', 'spectral_image.mat'))['img']
    cube = sio.loadmat('C:\Roman\HDSP_Activities\DeepOptix\deepoptix\examples\data\spectral_image.mat')['img']
    # load optical encoder
    cube_tf = tf.convert_to_tensor(cube)[None]  # None add a new dimension
    print(cube_tf.shape)
    model = Autoencoder(reduce_spatial=True,l_ch=3,num_layers=5,dim_last = cube.shape[-1],num_filt=8,kernel_size=3)
    model.build(cube_tf.shape)  # this is only for the demo

    # encode the cube

    recon = model(cube_tf)

    # Print information about tensors


    # visualize the measurement

    plt.figure(figsize=(10, 10))

    plt.subplot(211)
    plt.title('cube')
    plt.imshow(cube[..., 0])

    plt.subplot(212)
    plt.title('recon')
    plt.imshow(recon[0, ..., 0])

    plt.show()
