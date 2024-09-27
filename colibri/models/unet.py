""" Unet Architecture """

from . import custom_layers
import torch.nn as nn


class Unet(nn.Module):
    r"""
    Unet Model

    The U-Net model is a fully convolutional neural network initially proposed for biomedical image segmentation.
    Similar to the autoencoder model, the U-Net model consists of an encoder and a decoder.
    The encoder extracts features from the input image, while the decoder upsamples these features to the original image size.
    During the upsampling process, the decoder uses skip connections to concatenate features from the encoder with the upsampled features.
    These skip connections help preserve the spatial information of the input image, which is the key difference between the U-Net and the autoencoder model.

    Implementation based on the formulation of authors in https://doi.org/10.1007/978-3-319-24574-4_28

    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=[32, 64, 128, 256],
        last_activation="sigmoid",
        **kwargs,
    ):
        r"""

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            features (list, optional): number of features in each level of the Unet. Defaults to [32, 64, 128, 256].
            last_activation (str, optional): activation function for the last layer. Defaults to 'sigmoid'.

        Returns:
            torch.nn.Module: Unet model

        """

        super().__init__()

        levels = len(features)

        self.inc = custom_layers.convBlock(in_channels, features[0], mode="CBRCBR")

        # -----------------  Down Path ----------------- #
        self.downs = nn.ModuleList(
            [
                custom_layers.downBlock(features[i], features[i + 1])
                for i in range(len(features) - 2)
            ]
        )

        # -----------------  Bottleneck  ----------------- #
        self.bottle = custom_layers.downBlock(features[-2], features[-1])

        # -----------------  Up Path ----------------- #
        self.ups = nn.ModuleList(
            [custom_layers.upBlock(features[i]) for i in range(len(features) - 2, 0, -1)]
            + [custom_layers.upBlock(features[0])]
        )

        # -----------------  Output ----------------- #
        self.outc = custom_layers.outBlock(features[0], out_channels, last_activation)

    def forward(self, x):
        r"""
        Args :

            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
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


class Unet_KD(Unet):
    r"""
    Unet Model that returns intermediate features
    """

    def forward(self, x):
        r"""
        Args :

            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        outputs = []
        feats = []

        x = self.inc(x)

        outputs.append(x)
        feats.append(x)

        for down in self.downs:
            x = down(x)
            outputs.append(x)
            feats.append(x)

        x = self.bottle(x)

        for up in self.ups:
            x = up(x, outputs.pop())
            feats.append(x)

        return self.outc(x), feats
