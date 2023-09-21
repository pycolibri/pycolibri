""" Unet Architecture """

from . import custom_layers
import tensorflow as tf
class Unet(tf.keras.layers.Layer):
    """
    Unet Layer

    """

    def __init__(self, 
                 out_channels, 
                 features=[32, 64, 128, 256],
                 last_activation='sigmoid'):
        """ Unet Layer

        Args:
            out_channels (int): number of output channels
            features (list, optional): number of features in each level of the Unet. Defaults to [32, 64, 128, 256].
            last_activation (str, optional): activation function for the last layer. Defaults to 'sigmoid'.
        
        Returns:
            tf.keras.Layer: Unet model
            
        """
        
        super(Unet, self).__init__()

        levels = len(features)

        self.inc = custom_layers.convBlock(features[0], mode='CBRCBR') 

        self.downs = [
            custom_layers.downBlock(features[i+1]) for i in range(levels-2)
        ]

        self.bottle = custom_layers.downBlock(features[-1] // 2)

        self.ups = [
            custom_layers.upBlock(features[i] // 2) for i in range(levels-2, 0, -1)
        ]

        self.ups.append(custom_layers.upBlock(features[0] // 2))


        self.outc = custom_layers.outBlock(out_channels, last_activation)


    def call(self, x):

        outputs = []
        x = self.inc(x)
        outputs.append(x)
        
        for down in self.downs:
            x = down(x)
            outputs.append(x)

        x = self.bottle(x)

        for up in self.ups:
            x = up(x, outputs.pop())
        
        return self.outc(x)