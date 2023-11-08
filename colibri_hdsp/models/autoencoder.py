""" Autoencoder Architecture """
from . import custom_layers
import torch.nn as nn

class Autoencoder(nn.Module):
    """
    Autoencoder layer
    """

    def __init__(self, 
                 in_channels=1,
                 out_channels=1, 
                 features=[32, 64, 128, 256],
                 last_activation='sigmoid',
                 reduce_spatial = False):
        """ Autoencoder Layer

            Args:
                out_channels (int): number of output channels
                features (list, optional): number of features in each level of the Unet. Defaults to [32, 64, 128, 256].
                last_activation (str, optional): activation function for the last layer. Defaults to 'sigmoid'.
                reduce_spatial (bool): select if the autoencder reduce spatial dimension
                
            
            Returns:
                torch.nn.Module: Autoencoder model
                
        """    
        super(Autoencoder, self).__init__()

        levels = len(features)

        self.inc = custom_layers.convBlock(in_channels,features[0], mode='CBRCBR') 
        if reduce_spatial:
            self.downs = nn.ModuleList(
                [
                    custom_layers.downBlock(features[i], features[i + 1])
                    for i in range(len(features) - 1)
                ]
            )

            self.ups = nn.ModuleList(
                [
                    custom_layers.upBlockNoSkip(features[i+1],features[i])
                    for i in range(len(features)-2, 0, -1)
                ]
                + [custom_layers.upBlockNoSkip(features[1],features[0])]
            )
            # self.ups.append(custom_layers.upBlockNoSkip(features[0]))
            self.bottle = custom_layers.convBlock(features[-1], features[-1])

        else:
            self.downs =  nn.ModuleList([
                custom_layers.convBlock(features[i],features[i+1], mode='CBRCBR') 
                for i in range(levels-1)
            ])

            self.bottle = custom_layers.convBlock(features[-1], features[-1])

            self.ups = nn.ModuleList(
                [
                    custom_layers.convBlock(features[i+1],features[i])
                    for i in range(len(features) - 2, 0, -1)
                ]
                + [custom_layers.convBlock(features[1],features[0])]
            )
            
        self.outc = custom_layers.outBlock(features[0], out_channels, last_activation)

    def forward(self, inputs, get_latent=False,**kwargs):

        x = self.inc(inputs)
        
        for down in self.downs:
            print('d',x.shape)
            x = down(x)

        xl = self.bottle(x)
        x = xl
        print('b',x.shape)

        for up in self.ups:
            print('u',x.shape)

            x = up(x)
        if get_latent:
            return self.outc(x),xl
        else:
            return self.outc(x)
        

