""" Unet Architecture """

import tensorflow as tf
import layers as custom_layers

class Unet(tf.keras.layers.Layer):
    """
    Unet Layer

    :param out_channels: number of output channels
    :param in_features: number of input features
    :param levels: number of levels in the Unet
    :param last_activation: activation function for the last layer

    :return: Unet model
    :rtype: tf.keras.Layer
    """


    def __init__(self, 
                 out_channels, 
                 in_features=32, 
                 levels=4, 
                 last_activation='sigmoid'):
        
        super(Unet, self).__init__()

        features = [in_features * 2**i for i in range(levels)]

        self.inc = custom_layers.convBlock(features[0]) 

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


def unet(in_channels, out_channels, size=None):

    _input = tf.keras.Input(shape=(size, size, in_channels))
    output = Unet(out_channels)(_input)
    model = tf.keras.Model(inputs=_input, outputs=output)

    return model


if __name__ == "__main__":

    import numpy as np

    size = 128
    in_channels = 16
    out_channels = 1

    model = unet(in_channels, out_channels, size)

    x = np.random.randn(1, size, size, in_channels).astype(np.float32)
    y = model(x)

    print("input shape: ", x.shape)
    print("output shape: ", y.shape)

    # model summary
    model.summary()

