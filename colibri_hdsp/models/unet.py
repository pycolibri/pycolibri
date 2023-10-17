""" Unet Architecture """

from . import custom_layers
# import tensorflow as tf
import torch
import torch.nn as nn

class Unet(nn.Module):
    """
    Unet Layer

    """

    def __init__(self, 
                 in_channels=1,
                 out_channels=1, 
                 features=[32, 64, 128, 256],
                 last_activation=nn.Sigmoid):
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

        self.inc = custom_layers.convBlock(in_channels, features[0], mode='CBRCBR') 
        
        # -----------------  Down Path ----------------- #
        self.downs = ()
        for i in range(levels-2):
            self.downs += (custom_layers.downBlock(features[i], features[i+1]),)
        self.downs = nn.ModuleList(self.downs)


        # -----------------  Bottleneck  ----------------- #
        self.bottle = custom_layers.downBlock(features[-2], features[-1] )


        # -----------------  Up Path ----------------- #
        self.ups = ()
        for i in range(levels-2, 0, -1):
            self.ups += (custom_layers.upBlock(features[i]), )

        self.ups += (custom_layers.upBlock(features[0]), )
        self.ups = nn.ModuleList(self.ups)

        
        # -----------------  Output ----------------- #
        self.outc = custom_layers.outBlock(features[0],  out_channels, last_activation)


    def forward(self, x):

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