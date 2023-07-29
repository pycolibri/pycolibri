import tensorflow as tf
import numpy as np


class CASSI(tf.keras.layers.Layer):
    def __init__(self, trainable=False, ca_regularizer=None, initial_ca=None, seed=None):
        """
        Layer that performs the forward and transpose operator of coded aperture snapshot spectral imager (CASSI), more information refer to: Compressive Coded Aperture Spectral Imaging: An Introduction: https://doi.org/10.1109/MSP.2013.2278763
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

    def build(self, input_shape):
        super(CASSI, self).build(input_shape)
        self.M, self.N, self.L = input_shape  # Extract spectral image shape

        if self.initial_ca is None:
            initializer = tf.random_uniform_initializer(minval=0, maxval=1, seed=self.seed)
        else:
            assert self.initial_ca.shape != (1, self.M, self.N, 1), "the start CA shape should be (1, M, N, 1)"
            initializer = tf.constant_initializer(self.initial_ca)

        self.ca = self.add_weight(name='coded_apertures', shape=(1, self.M, self.N, 1), initializer=initializer,
                                  trainable=self.trainable, regularizer=self.ca_regularizer)

    def forward(self, x, ca=None):  # perform H * x
        ca = self.ca if ca is None else ca
        y1 = tf.multiply(x, ca)  # Multiplication of the scene by the coded aperture

        # shift and sum
        y2 = tf.zeros((1, self.M, self.N + self.L - 1, 1))  # Variable that will serve as the measurement
        for l in range(self.L):
            # Shifting produced by the prism 
            y2 += tf.pad(y1[..., l, None], [(0, 0), (0, 0), (l, self.L - l - 1), (0, 0)])

        return y2

    def backward(self, y, ca=None):  # perform H^T * y
        ca = self.ca if ca is None else ca
        x = tf.concat([y[..., l:l + self.N, :] for l in range(self.L)],
                      axis=-1)  # Undo unshifting and create cube version of measurement
        return tf.multiply(x, ca)

    def forward_backward(self, x, ca=None):  # perform H^T * H * x
        ca = self.ca if ca is None else ca
        return self.backward(self.forward(x, ca), ca)

    def backward_forward(self, y, ca=None):  # perform H * H^T * y
        ca = self.ca if ca is None else ca
        return self.forward(self.backward(y, ca), ca)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import scipy.io as sio
    import os

    # load a mat file

    cube = sio.loadmat(os.path.join('examples', 'data', 'spectral_image.mat'))['img']  # (M, N, L)
    ca = np.random.rand(1, cube.shape[0], cube.shape[1], 1)  # custom ca (1, M, N, 1)

    # load optical encoder

    cassi = CASSI()
    cassi.build(cube.shape)  # this is only for the demo

    # encode the cube

    cube_tf = tf.convert_to_tensor(cube)[None]  # None add a new dimension
    measurement = cassi.forward(cube_tf, ca)
    transpose = cassi.backward(measurement, ca)
    direct_transpose = cassi.forward_backward(cube_tf, ca)
    measurement2 = cassi.forward(transpose, ca)

    # Print information about tensors

    print('cube shape: ', cube_tf.shape)
    print('measurement shape: ', measurement.shape)
    print('transpose shape: ', transpose.shape)

    # visualize the measurement

    plt.figure(figsize=(10, 10))

    plt.subplot(221)
    plt.title('cube')
    plt.imshow(cube[..., 0])

    plt.subplot(222)
    plt.title('measurement')
    plt.imshow(measurement[0, ..., 0])

    plt.subplot(223)
    plt.title('transpose')
    plt.imshow(transpose[0, ..., 0])

    plt.subplot(224)
    plt.title('measurement2')
    plt.imshow(measurement2[0, ..., 0])

    plt.tight_layout()
    plt.show()
