import torch
from functional import psf_single_doe_spectral, convolutional_sensing, weiner_filter
from sota_does import fresnel_lens
from utils import BaseOpticsLayer

class SingleDOESpectral(BaseOpticsLayer):
    r"""
    Single Diffractive Optical Element for Spectral Imaging 

    in progress

    """

    def __init__(self, input_shape, 
                        height_map, 
                        aperture, 
                        wavelengths, 
                        source_distance, 
                        sensor_distance, 
                        pixel_size,
                        trainable = False):
        r"""
        Initializes the SingleDOESpectral layer.

        Args:
            input_shape (tuple): Tuple, shape of the input image (L, M, N).
            trainable (bool): Boolean, if True the coded aperture is trainable
            initial_ca (torch.Tensor): Initial coded aperture with shape (1, 1, M, N)
        """
        
        self.trainable = trainable
        self.aperture = aperture
        self.wavelengths = wavelengths
        self.source_distance = source_distance
        self.sensor_distance = sensor_distance
        self.pixel_size = pixel_size


        self.L, self.M, self.N = input_shape  # Extract spectral image shape

        
        if height_map is None:
            height_map = fresnel_lens(ny=self.M, nx=self.N, focal=1, radius=1)
        #Add parameter CA in pytorch manner
        self.height_map = torch.nn.Parameter(height_map, requires_grad=self.trainable)

        super(SingleDOESpectral, self).__init__(learnable_optics=self.height_map, sensing=self.sensing, backward=self.backward)

    def sensing(self, x):
        r"""
        Forward operator of the SingleDOESpectral layer.

        Args:
            x (torch.Tensor): Input tensor with shape (B, L, M, N)
        Returns:
            torch.Tensor: Output tensor with shape (B, 1, M, N) 
        """

        psf = psf_single_doe_spectral(height_map=self.height_map, 
                                        aperture=self.aperture, 
                                        wavelengths=self.wavelengths,
                                        source_distance=self.source_distance,
                                        sensor_distance=self.sensor_distance,
                                        pixel_size=self.pixel_size)
        
        return convolutional_sensing(x, psf)

    def backward(self, x):
        r"""
        Backward operator of the SingleDOESpectral layer.

        Args:
            x (torch.Tensor): Input tensor with shape (B, 1, M, N)
        Returns:
            torch.Tensor: Output tensor with shape (B, L, M, N) 
        """

        psf = psf_single_doe_spectral(height_map=self.height_map, 
                                        aperture=self.aperture, 
                                        wavelengths=self.wavelengths,
                                        source_distance=self.source_distance,
                                        sensor_distance=self.sensor_distance,
                                        pixel_size=self.pixel_size)
        
        return weiner_filter(x, psf)
    def forward(self, x, type_calculation="forward"):
        r"""
        Performs the forward or backward operator according to the type_calculation

        Args:
            x (torch.Tensor): Input tensor with shape (B, L, M, N)
            type_calculation (str): String, it can be "forward", "backward" or "forward_backward"
        Returns:
            torch.Tensor: Output tensor with shape (B, L, M, N) 
        Raises:
            ValueError: If type_calculation is not "forward", "backward" or "forward_backward"
        """

        return super(SingleDOESpectral, self).forward(x, type_calculation)
