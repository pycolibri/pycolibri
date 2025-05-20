""" Utilities for building layers. """


import torch
import torch.nn as nn


class Activation(nn.Module):
    """Activation Layer"""
    def __init__(self, activation="relu"):
        super(Activation, self).__init__()
        """ Activation Layer
        
        Args:
            activation (str or nn.functional, optional): Activation function, such as tf.nn.relu, or string name of built-in activation function, such as "relu".
        
        Returns:
            nn.Module: Activation layer
        """

        if isinstance(activation, str):
            self.act_fn = self.get_activation(activation)
        else:
            self.act_fn = activation

    def get_activation(self, name):
        """
        Get activation function by name.
        
        Args:
            name (str): Name of the activation function.
        
        Returns:
            nn.Module: Activation function
        """

        activations = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "softmax": nn.Softmax(dim=1),
            "tanh": nn.Tanh(),
            "identity": nn.Identity(),
        }

        if name in activations.keys():
            return activations[name]
        else:
            raise ValueError(f"Unknown activation function: {name}")
        
    def forward(self, x):
        """
        Computes the activation function.
        
        Args: 
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.act_fn(x)

class convBlock(nn.Module):
    """Convolutional Block

    default configuration: (Conv2D => Batchnorm => ReLU) * 2
    
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        bias=False,
        mode="CBR",
        factor=2,
    ):
        """Convolutional Block

        Args:
            out_channels (int, optional): number of output channels. Defaults to 1.
            kernel_size (int, optional): size of the kernel. Defaults to 3.
            bias (bool, optional): whether to use bias or not. Defaults to False.
            mode (str, optional): mode of the convBlock, posible values are: ['C', 'B', 'R', 'U', 'M', 'A']. Defaults to 'CBR'.
            factor (int, optional): factor for upsampling/downsampling. Defaults to 2.

        """

        super(convBlock, self).__init__()

        self.layers = nn.ModuleList()
        
        pad_size = kernel_size // 2

        conv_kwargs = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=pad_size,
            bias=bias,
        )
        first_conv = True

        for c in mode:
            layer = self.build_layer(c, conv_kwargs, factor)
            self.layers.append(layer)

            if c == "C" and first_conv:
                first_conv = False
                conv_kwargs["in_channels"] = conv_kwargs["out_channels"]

    def forward(self, x):
        """
        Forward pass of the convBlock.

        Args:

            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output tensor
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def build_layer(self, c, params, factor):
        """
        Build layer based on the mode.

        Args:
            c (str): mode of the layer
            params (dict): parameters for the layer
            factor (int): factor for upsampling/downsampling
        
        Returns:
            nn.Module: Layer
        """
        num_features = params["out_channels"]
        batch_norm_params = dict(num_features=num_features)

        params_mapping = {
            "C": (nn.Conv2d, params),
            "B": (nn.BatchNorm2d, batch_norm_params),
            "R": (nn.ReLU, None),
            "U": (nn.Upsample, dict(size=(factor, factor))),
            "M": (nn.MaxPool2d, dict(kernel_size=(factor, factor))),
            "A": (nn.AvgPool2d, dict(kernel_size=(factor, factor))),
        }

        if c in params_mapping.keys():
            layer, params = params_mapping[c]
            return layer(**params) if params else layer()
        else:
            raise ValueError(f"Unknown layer type: {c}")


class downBlock(nn.Module):
    """Spatial downsampling and then convBlock"""

    def __init__(self, in_channels, out_channels):

        """
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
        
        Returns:
            nn.Module: DownBlock model
        """
        super(downBlock, self).__init__()

        self.pool_conv = convBlock(in_channels, out_channels, mode="MCBRCBR")

    def forward(self, x):
        return self.pool_conv(x)


class upBlock(nn.Module):
    """Spatial upsampling and then convBlock"""

    def __init__(self, in_channels):
        """
        Args:
            in_channels (int): number of input channels
        
        """
        super(upBlock, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
        )

        self.conv_block = nn.Sequential(
            convBlock(in_channels * 2, in_channels), convBlock(in_channels, in_channels)
        )

    def forward(self, x1, x2):
        """
        Forward pass of the upBlock.

        Args:   
            x1 (torch.Tensor): Input tensor
            x2 (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output tensor
        """
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.shape[-2] - x1.shape[-2]
        diffX = x2.shape[-1] - x1.shape[-1]

        x1 = nn.functional.pad(
            x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2)
        )

        return self.conv_block(torch.cat([x2, x1], dim=1))


class upBlockNoSkip(nn.Module):
    """Spatial upsampling and then convBlock"""

    def __init__(self, in_channels,out_channels):
        """
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
        
        """
        super(upBlockNoSkip, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv_block = nn.Sequential(
            convBlock(in_channels,out_channels ), convBlock(out_channels, out_channels)
        )

    def forward(self, x1):
        """
        Forward pass of the upBlock.

        Args:
            x1 (torch.Tensor): Input tensor

        Returns:

            torch.Tensor: Output tensor
        """
        x1 = self.up(x1)
        # input is CHW
        return self.conv_block(x1)


class outBlock(nn.Module):
    """Convolutional Block with 1x1 kernel and without activation"""

    def __init__(self, in_channels, out_channels, activation=None):
        """
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            activation (str, optional): activation function. Defaults to None.
        """
        super(outBlock, self).__init__()

        self.conv = convBlock(in_channels, out_channels, kernel_size=1, mode="C")
        self.act = Activation(activation) if activation else nn.Identity()

    def forward(self, x):
        """
        Forward pass of the outBlock.

        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output tensor
        """
        x = self.conv(x)
        return self.act(x)
