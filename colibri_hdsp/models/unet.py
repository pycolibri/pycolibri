""" Unet Architecture """

from . import custom_layers

# import tensorflow as tf
import torch
import torch.nn as nn


class Unet(nn.Module):
    """
    Unet Layer

    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=[32, 64, 128, 256],
        last_activation="sigmoid",
    ):
        """Unet Layer

        Args:
            out_channels (int): number of output channels
            features (list, optional): number of features in each level of the Unet. Defaults to [32, 64, 128, 256].
            last_activation (str, optional): activation function for the last layer. Defaults to 'sigmoid'.

        Returns:
            tf.keras.Layer: Unet model

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
            [
                custom_layers.upBlock(features[i])
                for i in range(len(features) - 2, 0, -1)
            ]
            + [custom_layers.upBlock(features[0])]
        )

        # -----------------  Output ----------------- #
        self.outc = custom_layers.outBlock(features[0], out_channels, last_activation)

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
