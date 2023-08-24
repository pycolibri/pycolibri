""" Autoencoder Architecture """

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import layers as custom_layers


# TODO: Implement Autoencoder model

class Autoencoder(Layer):
    """
    Autoencoder layer
    """

    def __init__(self, 
                 out_channels, 
                 features=[32, 64, 128, 256],
                 last_activation='sigmoid',
                 reduce_spatial = False):
        """ Unet Layer

            Args:
                out_channels (int): number of output channels
                features (list, optional): number of features in each level of the Unet. Defaults to [32, 64, 128, 256].
                last_activation (str, optional): activation function for the last layer. Defaults to 'sigmoid'.
                reduce_spatial (bool): select if the autoencder reduce spatial dimension
                
            
            Returns:
                tf.keras.Layer: Unet model
                
        """    
        super(Autoencoder, self).__init__()

        levels = len(features)

        self.inc = custom_layers.convBlock(features[0], mode='CBRCBR') 
        if reduce_spatial:
            self.downs = [
                custom_layers.downBlock(features[i+1]) for i in range(levels-2)
            ]
            self.ups = [
                custom_layers.upBlockNoSkip(features[i] // 2) for i in range(levels-2, 0, -1)
            ]
            self.bottle = custom_layers.downBlock(features[-1] // 2)
            self.ups.append(custom_layers.upBlockNoSkip(features[0] // 2))
        else:
            self.downs = [
                custom_layers.convBlock(features[i+1], mode='CBRCBR') for i in range(levels-2)
            ]

            self.bottle = custom_layers.convBlock(features[-1], mode='CBRCBR')

            self.ups = [
                custom_layers.convBlock(features[i] // 2, mode='CBRCBR') for i in range(levels-2, 0, -1)
            ]
            self.ups.append(custom_layers.convBlock(features[0],mode='CBCBR'))

            
        self.outc = custom_layers.outBlock(out_channels, last_activation)

    def call(self, inputs, get_latent=False,**kwargs):

        x = self.inc(inputs)
        
        for down in self.downs:
            x = down(x)

        xl = self.bottle(x)
        x = xl
        for up in self.ups:
            x = up(x)
        if get_latent:
            return self.outc(x),xl
        else:
            return self.outc(x)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import scipy.io as sio
    import os

    # load a mat file

    cube = sio.loadmat(os.path.join('deepoptix', 'examples', 'data', 'spectral_image.mat'))['img']  # (M, N, L)


    # load optical encoder
    cube_tf = tf.convert_to_tensor(cube)[None]  # None add a new dimension
    print(cube_tf.shape)
    model = Autoencoder(reduce_spatial=True,out_channels=cube_tf.shape[-1],features=[4,16,6],last_activation='relu')
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
