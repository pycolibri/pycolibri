""" Unet Architecture """

import tensorflow as tf
import layers as custom_layers

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

