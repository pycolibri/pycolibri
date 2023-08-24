
import tensorflow as tf
import tensorflow.keras.layers as layers
from autoencoder import *
from unet import *
import tensorflow as tf


class DIP_CASSI(tf.keras.models.Model):
    """
    DIP CASSI Model

    """

    def __init__(self,input_shape=[128,128,25], recon='unet',initilization='random',initial_ca=None,mode='base',network_args=None,):
        """ DIP Model

        Args:
            input_shape (list): shape of the input
            recon (str): reconstruction network
            initialization (str): type of initilziation
            initial_ca (array): coded aperture
            mode (str):  cassi mode

        
        Returns:
            tf.keras.model: DIP Model
            
        """
        super(DIP_CASSI,self).__init__()
        self.initilization = initilization
        self.optics = CASSI(mode=mode, initial_ca=initial_ca)
        self.input_shape = input_shape
        if recon=='unet':
            self.recon = Unet(out_channels=input_shape[-1],features=network_args['features'],last_activation='relu')
        elif recon == 'Autoencoder':
             self.recon = Autoencoder(out_channels=input_shape[-1],features=network_args['features'],last_activation='relu',reduce_spatial=network_args['reduce_spatial'])   
        else:
            raise ValueError("Choose models autoencoder or unet")

    def __call__(self,y,is_training=True):
        if self.initilization=='random':
            z0 = tf.random.normal(shape=(1,*self.input_shape))
        if self.initilization=='transpose':
            z0 = self.optics(y,type_calculation='backward')

        
        x = self.recon(z0)

        ys = self.optics(x,type_calculation='forward')

        if is_training:
            return ys
        else:
            return x
        



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import scipy.io as sio
    
    from deepoptix.optics.cassi import CASSI
    import os

    # load a mat file

    cube = sio.loadmat(os.path.join('deepoptix', 'examples', 'data', 'spectral_image.mat'))['img']  # (M, N, L)

    ca = np.random.rand(1, cube.shape[0], cube.shape[1], 1)  # custom ca (1, M, N, 1)

    mode = 'base'
    cassi = CASSI(mode,initial_ca=ca)

    cube_tf = tf.convert_to_tensor(cube)[None]  # None add a new dimension
    y = cassi(cube_tf, type_calculation="forward")
    network_args = {'features': [28,32,14],'reduce_spatial':False,'out_channels':cube_tf.shape[-1],}
    model = DIP_CASSI(input_shape=cube.shape,recon='autoencoder',initilization='random',network_args=network_args)


    # load optical encoder
    

    recon = model(y,True)


    # Print information about tensors


    # visualize the measurement

    plt.figure(figsize=(10, 10))


    plt.title('cube')
    plt.imshow(y[..., 0])



    plt.show()
