from unet import Unet

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

models = {
    "Unet": Unet
}

__all__ = models.keys()

def build_network(model_name="Unet", in_channels=1, out_channels=1, size=None, last_activation='sigmoid'):
    """ Build the network model

    Args:
        model_name (str, optional): Model name. Defaults to "unet".
        in_channels (int, optional): Number of input channels. Defaults to 1.
        out_channels (int, optional): Number of output channels. Defaults to 1.
        size (int, optional): Size of the input image. Defaults to None.
        last_activation (str, optional): Activation function for the last layer. Defaults to 'relu'.

    """

    if model_name not in models.keys():
        raise ValueError(f"Model {model_name} not found. Available models: {list(models.keys())}")
    
    _input = Input(shape=(size, size, in_channels))
    output = models[model_name](out_channels, last_activation=last_activation)(_input)
    model = Model(inputs=_input, outputs=output)

    return model
    
if __name__ == "__main__":
    import tensorflow as tf

    model = build_network(model_name="Unet", in_channels=1, out_channels=1, size=32)
    model.summary()

    x = tf.random.normal((1, 32, 32, 1))
    y = model(x)
    
    print("input shape:", x.shape)
    print("output shape:", y.shape)






