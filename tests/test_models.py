from test_utils import include_colibri
include_colibri()

from colibri_hdsp.models import build_network, Autoencoder, Unet


if __name__ == "__main__":
    import tensorflow as tf


    model_layer = Autoencoder
    model = build_network(model=model_layer, size=32, in_channels=1, out_channels=1)
    model.summary()

    x = tf.random.normal((1, 32, 32, 1))
    y = model(x)
    
    print("input shape:", x.shape)
    print("output shape:", y.shape)

