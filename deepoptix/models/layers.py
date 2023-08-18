""" Utilities for building layers. """

import tensorflow as tf
import tensorflow.keras.layers as layers


class convBlock(layers.Layer):
    """ Convolutional Block

    default configuration: (Conv2D => Batchnorm => ReLU) * 2

    """

    def __init__(self, out_channels=1, kernel_size=3, bias=False, mode='CBR'):
        """ Convolutional Block

        Args:
            out_channels (int, optional): number of output channels. Defaults to 1.
            kernel_size (int, optional): size of the kernel. Defaults to 3.
            bias (bool, optional): whether to use bias or not. Defaults to False.
            mode (str, optional): mode of the convBlock, posible values are: ['C', 'B', 'R', 'U', 'M', 'A']. Defaults to 'CBR'.
            
        """


        super(convBlock, self).__init__()

        self.layers = []
        conv_kwargs = dict(filters=out_channels, 
                           kernel_size=kernel_size, 
                           padding='same', 
                           use_bias=bias)

        for c in mode:
            layer = self.build_layer(c, conv_kwargs)
            self.layers.append(layer)
        
    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def build_layer(self, c, params):

        params_mapping = {
            'C': (layers.Conv2D, params),
            'B': (layers.BatchNormalization, None),
            'R': (layers.ReLU, None),
            'U': (layers.UpSampling2D, dict(size=(2,2))),
            'M': (layers.MaxPool2D, dict(pool_size=(2,2))),
            'A': (layers.AveragePooling2D, dict(pool_size=(2,2))),
        }

        if c in params_mapping.keys():
            layer, params = params_mapping[c]
            return layer(**params) if params else layer()
        else:
            raise ValueError(f'Unknown layer type: {c}')




class downBlock(layers.Layer):
    """Spatial downsampling and then convBlock"""

    def __init__(self, out_channels):
        super(downBlock, self).__init__()

        self.pool_conv = convBlock(out_channels, mode='MCBRCBR')

    def call(self, x):
        return self.pool_conv(x)


class upBlock(layers.Layer):
    """Spatial upsampling and then convBlock"""

    def __init__(self, out_channels):
        super(upBlock, self).__init__()

        self.up = layers.UpSampling2D(size=2, interpolation='bilinear')
        self.conv_block = tf.keras.Sequential([
            convBlock(out_channels // 2),
            convBlock(out_channels)
        ])


    def call(self, x1, x2):
        
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.shape[1] - x1.shape[1]
        diffX = x2.shape[2] - x1.shape[2]

        x1 = tf.pad(x1, [[0, 0], [diffX // 2, diffX - diffX // 2],
                         [diffY // 2, diffY - diffY // 2], [0, 0]])

        return self.conv_block(tf.concat([x2, x1], axis=-1))


class outBlock(layers.Layer):
    """Convolutional Block with 1x1 kernel and without activation"""

    def __init__(self, out_channels, activation=None):
        super(outBlock, self).__init__()

        self.conv = convBlock(out_channels, kernel_size=1, mode='C')
        self.act  = layers.Activation(activation) if activation else None

    def call(self, x):
        x = self.conv(x)
        return self.act(x) if self.act else x

