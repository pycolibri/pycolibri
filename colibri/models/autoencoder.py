""" Autoencoder Architecture """

from . import custom_layers
import torch.nn as nn


class Autoencoder(nn.Module):
    r"""
    Autoencoder Model

    The autoencoder model is a neural network that is trained to learn a latent representation of the input data. The model is composed of an encoder and a decoder. The encoder compresses the input data into a latent space representation, while the decoder reconstructs the input data from the latent space representation. Usually, the autoencoder model is trained to minimize the reconstruction error between the input data and the reconstructed data as follows:

    .. math::
        \mathcal{L} = \left\| \mathbf{x}- D(E(\mathbf{x})) \right\|_2^2

    where :math:`\mathbf{x}` is the input data and :math:`\hat{\mathbf{x}} = D(E(\mathbf{x}))` is the reconstructed data with :math:`E(\cdot)` and :math:`D(\cdot)` the encoder and decoder networks, respectively.
   
    Implementation based on the formulation of authors in https://dl.acm.org/doi/book/10.5555/3086952
    
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=[32, 64, 128, 256],
        last_activation="sigmoid",
        reduce_spatial=False,
        **kwargs,
    ):
        r"""

        Args:

            in_channels (int): number of input channels
            out_channels (int): number of output channels
            features (list, optional): number of features in each level of the Unet. Defaults to [32, 64, 128, 256].
            last_activation (str, optional): activation function for the last layer. Defaults to 'sigmoid'.
            reduce_spatial (bool): select if the autoencder reduce spatial dimension


        Returns:
            torch.nn.Module: Autoencoder model

        """
        super(Autoencoder, self).__init__()

        levels = len(features)

        self.inc = custom_layers.convBlock(in_channels, features[0], mode="CBRCBR")
        if reduce_spatial:
            self.downs = nn.ModuleList(
                [
                    custom_layers.downBlock(features[i], features[i + 1])
                    for i in range(len(features) - 1)
                ]
            )

            self.ups = nn.ModuleList(
                [
                    custom_layers.upBlockNoSkip(features[i + 1], features[i])
                    for i in range(len(features) - 2, 0, -1)
                ]
                + [custom_layers.upBlockNoSkip(features[1], features[0])]
            )
            # self.ups.append(custom_layers.upBlockNoSkip(features[0]))
            self.bottle = custom_layers.convBlock(features[-1], features[-1])

        else:
            self.downs = nn.ModuleList(
                [
                    custom_layers.convBlock(features[i], features[i + 1], mode="CBRCBR")
                    for i in range(levels - 1)
                ]
            )

            self.bottle = custom_layers.convBlock(features[-1], features[-1])

            self.ups = nn.ModuleList(
                [
                    custom_layers.convBlock(features[i + 1], features[i])
                    for i in range(len(features) - 2, 0, -1)
                ]
                + [custom_layers.convBlock(features[1], features[0])]
            )

        self.outc = custom_layers.outBlock(features[0], out_channels, last_activation)

    def forward(self, inputs, get_latent=False, **kwargs):
        r"""
        Forward pass of the autoencoder

        Args:
            inputs (torch.Tensor): input tensor
            get_latent (bool): if True, return the latent space
        
        Returns:
            torch.Tensor: output tensor
        """

        x = self.inc(inputs)

        for down in self.downs:
            x = down(x)

        xl = self.bottle(x)
        x = xl

        for up in self.ups:
            x = up(x)

        if get_latent:
            return self.outc(x), xl
        else:
            return self.outc(x)
