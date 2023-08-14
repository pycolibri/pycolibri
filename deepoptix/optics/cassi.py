import tensorflow as tf
import numpy as np

def forward_cassi(x, ca):
    """
    Forward operator of coded aperture snapshot spectral imager (CASSI), more information refer to: Compressive Coded Aperture Spectral Imaging: An Introduction: https://doi.org/10.1109/MSP.2013.2278763
    :param x: Spectral image with shape (1, M, N, L)
    :param ca: Coded aperture with shape (1, M, N, 1)
    :return: Measurement with shape (1, M, N + L - 1, 1)
    """
    y1 = tf.multiply(x, ca) # Multiplication of the scene by the coded aperture 
    _, M, N, L = y1.shape # Extract spectral image shape
    # shift and sum
    y2 = tf.zeros((1, M, N + L - 1, 1)) # Variable that will serve as the measurement
    for l in range(L):
        # Shifting produced by the prism 
        y2 += tf.pad(y1[..., l, None], [(0, 0), (0, 0), (l, L - l - 1), (0, 0)])

    return y2

def backward_cassi(y, ca):
    """
    Backward operator of coded aperture snapshot spectral imager (CASSI), more information refer to: Compressive Coded Aperture Spectral Imaging: An Introduction: https://doi.org/10.1109/MSP.2013.2278763
    :param y: Measurement with shape (1, M, N + L - 1, 1)
    :param ca: Coded aperture with shape (1, M, N, 1)
    :return: Spectral image with shape (1, M, N, L)
    """
    _, M, N, _ = y.shape # Extract spectral image shape
    L = N-M+1 # Number of shifts
    x = tf.concat([y[..., l:l + M, :] for l in range(L)], axis=-1) # Undo unshifting and create cube version of measurement
    return tf.multiply(x, ca)

class CASSI(tf.keras.layers.Layer):
    """
    Layer that performs the forward and backward operator of coded aperture snapshot spectral imager (CASSI), more information refer to: Compressive Coded Aperture Spectral Imaging: An Introduction: https://doi.org/10.1109/MSP.2013.2278763

    """
    def __init__(self, trainable=False, ca_regularizer = None, initial_ca = None, seed=None):
        """
        :param trainable: Boolean, if True the coded aperture is trainable
        :param ca_regularizer: Regularizer function applied to the coded aperture
        :param initial_ca: Initial coded aperture with shape (1, M, N, 1)
        :param seed: Random seed
        """
        super(CASSI, self).__init__(name='cassi')
        self.seed = seed
        self.trainable = trainable
        self.ca_regularizer = ca_regularizer
        self.initial_ca = initial_ca

        self.forward = forward_cassi
        self.backward = backward_cassi

    def build(self, input_shape):
        """
        Build method of the layer, it creates the coded aperture according to the input shape
        :param input_shape: Shape of the input tensor (1, M, N, L)
        :return: None
        """
        super(CASSI, self).build(input_shape)
        self.M, self.N, self.L = input_shape # Extract spectral image shape

        if self.initial_ca is None:
            initializer = tf.random_uniform_initializer(minval=0, maxval=1, seed=self.seed)
        else:
            assert self.initial_ca.shape != (1, self.M, self.N, 1), "the start CA shape should be (1, M, N, 1)"
            initializer = tf.constant_initializer(self.initial_ca)

        self.ca = self.add_weight(name='coded_apertures', shape=(1, self.M, self.N, 1), initializer=initializer,
                                    trainable=self.trainable, regularizer = self.ca_regularizer)

    def __call__(self, x, type_calculation = "forward"):
        """
        Call method of the layer, it performs the forward or backward operator according to the type_calculation
        :param x: Input tensor with shape (1, M, N, L)
        :param type_calculation: String, it can be "forward", "backward" or "forward_backward"
        :return: Output tensor with shape (1, M, N + L - 1, 1) if type_calculation is "forward", (1, M, N, L) if type_calculation is "backward, or (1, M, N, L) if type_calculation is "forward_backward
        :raises ValueError: If type_calculation is not "forward", "backward" or "forward_backward"
        """
        if type_calculation =="forward":
            return self.forward(x, self.ca)

        elif type_calculation =="backward":
            return self.backward(x, self.ca)
        elif type_calculation =="forward_backward":
            return self.backward(self.forward(x, self.ca), self.ca)
        
        else:
            raise ValueError("type_calculation must be forward, backward or forward_backward")
        
    

if __name__ == "__main__":


    import matplotlib.pyplot as plt
    import tensorflow as tf
    import scipy.io as sio
    import os

    # load a mat file

    cube = sio.loadmat(os.path.join('examples', 'data', 'spectral_image.mat'))['img']
    

    # load optical encoder

    cassi = CASSI()
    cassi.build(cube.shape)  # this is only for the demo

    # encode the cube

    cube_tf = tf.convert_to_tensor(cube)[None]  # None add a new dimension
    measurement = cassi(cube_tf, type_calculation="forward")
    backward = cassi(measurement, type_calculation="backward")
    direct_backward = cassi(cube_tf)
    measurement2 = cassi(backward, type_calculation="forward_backward")


    #Print information about tensors

    print('cube shape: ', cube_tf.shape)
    print('measurement shape: ', measurement.shape)
    print('backward shape: ', backward.shape)
    
    # visualize the measurement

    plt.figure(figsize=(10, 10))

    plt.subplot(221)
    plt.title('cube')
    plt.imshow(cube[..., 0])

    plt.subplot(222)
    plt.title('measurement')
    plt.imshow(measurement[0, ..., 0])

    plt.subplot(223)
    plt.title('backward')
    plt.imshow(backward[0, ..., 0])

    plt.subplot(224)
    plt.title('measurement2')
    plt.imshow(measurement2[0, ..., 0])

    plt.tight_layout()
    plt.show()
    