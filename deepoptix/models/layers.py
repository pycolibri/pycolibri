""" Utilities for building layers. """

import tensorflow as tf
import tensorflow.keras.layers as layers


class convBlock(layers.Layer):
    """(Conv2D => Batchnom => ReLU) * 2"""

    def __init__(self, out_channels, mid_channels=None):
        super(convBlock, self).__init__()

        if not mid_channels:
            mid_channels = out_channels

        conv_kwargs = dict(kernel_size=3, padding='same', use_bias=False)

        self.conv_block = tf.keras.Sequential([
            layers.Conv2D(mid_channels, **conv_kwargs),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(out_channels, **conv_kwargs),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])

    def call(self, x):
        return self.conv_block(x)
    

class downBlock(layers.Layer):
    """Spatial downsampling and then convBlock"""

    def __init__(self, out_channels):
        super(downBlock, self).__init__()

        self.pool_conv = tf.keras.Sequential([
            layers.MaxPool2D(2),
            convBlock(out_channels)
        ])

    def call(self, x):
        return self.pool_conv(x)


class upBlock(layers.Layer):
    """Spatial upsampling and then convBlock"""

    def __init__(self, out_channels):
        super(upBlock, self).__init__()

        self.up = layers.UpSampling2D(size=2, interpolation='bilinear')
        self.conv_block = convBlock(out_channels, out_channels // 2)


    def call(self, x1, x2):
        
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.shape[1] - x1.shape[1]
        diffX = x2.shape[2] - x1.shape[2]

        x1 = tf.pad(x1, [[0, 0], [diffX // 2, diffX - diffX // 2],
                         [diffY // 2, diffY - diffY // 2], [0, 0]])

        return self.conv_block(tf.concat([x2, x1], axis=-1))


class outBlock(layers.Layer):
    def __init__(self, out_channels, activation=None):
        super(outBlock, self).__init__()

        conv_kwargs = dict(kernel_size=1, padding='same', use_bias=False)

        self.conv = layers.Conv2D(out_channels, **conv_kwargs)
        self.act  = layers.Activation(activation) if activation else None

    def call(self, x):
        x = self.conv(x)
        return self.act(x) if self.act else x

