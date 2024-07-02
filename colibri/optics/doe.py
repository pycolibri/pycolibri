import torch
from colibri.optics.functional import psf_single_doe_spectral, convolutional_sensing, wiener_filter, ideal_panchromatic_sensor
from colibri.optics.sota_does import fresnel_lens, nbk7_refractive_index
from .utils import BaseOpticsLayer

class SingleDOESpectral(BaseOpticsLayer):
    r"""
    Single Diffractive Optical Element for Spectral Imaging 

    This optical system allow for the capture of spatio-spectral information through a wavelength dependent phase coding of light through a diffractive optical element. 
    
    Mathematically, this systems can be described as follows. 

    .. math::
    
            \mathbf{y} = \forwardLinear_{\learnedOptics}(\mathbf{x}) + \noise
    
    where :math:`\noise` is the sensor noise, :math:`\mathbf{x}\in\xset` is the input optical field, :math:`\mathbf{y}\in\yset` are the acquired signal, for SingleDOESpectral, :math:`\xset = \mathbb{R}^{L \times M \times N}` and :math:`\yset = \mathbb{R}^{M \times N}`, and :math:`\forwardLinear_{\learnedOptics}:\xset\rightarrow \yset` is the forward operator of the diffractive optical element, such as

    .. math::

        \begin{align*}
        \forwardLinear_{\learnedOptics}: \mathbf{x} &\mapsto \mathbf{y} \\
                        \mathbf{y}_l &=  \sum_{l=1}^{L} \psf(\learnedOptics)_l * \mathbf{x}_{, l}
        \end{align*}

    with :math:`\psf(\learnedOptics)_l` the optical-point spread function (PSF) of the diffractive optical element at wavelength :math:`\lambda_l`, calculated using  :func:`colibri.optics.functional.psf_single_doe_spectral`

    """
    def __init__(self, input_shape, 
                        height_map, 
                        aperture, 
                        wavelengths, 
                        source_distance, 
                        sensor_distance, 
                        pixel_size,
                        sensor_spectral_sensitivity=ideal_panchromatic_sensor,
                        doe_refractive_index = None,
                        approximation = "fresnel",
                        domain = "fourier",
                        trainable = False):
        r"""
        Initializes the SingleDOESpectral layer.



        Args:
            input_shape (tuple): The shape of the input spectral image in the format (L, M, N), where L is the number of spectral bands, M is the height, and N is the width.
            height_map (torch.Tensor): The height map of the DOE (Diffractive Optical Element). If None, a default fresnel lens will be generated.
            aperture (float): The aperture of the DOE.
            wavelengths (list): The list of wavelengths corresponding to the spectral bands.
            source_distance (float): The distance between the source and the DOE.
            sensor_distance (float): The distance between the DOE and the sensor.
            pixel_size (float): The size of a pixel in the sensor.
            sensor_spectral_sensitivity (torch.Tensor, optional): The spectral sensitivity of the sensor. Defaults to ideal_panchromatic_sensor.
            doe_refractive_index (float, optional): The refractive index of the DOE. Defaults to None.
            approximation (str, optional): The approximation used to calculate the PSF. It can be "fresnel", "angular_spectrum" or "fraunhofer". Defaults to "fresnel".
            domain (str, optional): The domain used to calculate the PSF. It can be "fourier" or "spatial". Defaults to "fourier".
            trainable (bool, optional): Whether the height map of the DOE is trainable or not. Defaults to False.
        """
        
        
        self.trainable = trainable
        self.aperture = aperture
        self.sensor_spectral_sensitivity = sensor_spectral_sensitivity
        self.wavelengths = wavelengths
        self.source_distance = source_distance
        self.sensor_distance = sensor_distance
        self.pixel_size = pixel_size
        self.approximation = approximation
        self.domain = domain

        self.refractive_index = nbk7_refractive_index if doe_refractive_index is None else doe_refractive_index


        self.L, self.M, self.N = input_shape  # Extract spectral image shape

        
        if height_map is None:
            height_map = fresnel_lens(ny=self.M, nx=self.N, focal=1, radius=1)
        #Add parameter CA in pytorch manner
        height_map = torch.nn.Parameter(height_map, requires_grad=self.trainable)

        super(SingleDOESpectral, self).__init__(learnable_optics=height_map, sensing=self.forward_convolution, backward=self.deconvolution)

    def get_psf(self, height_map=None):
        if height_map is None:
            height_map = self.learnable_optics
        return psf_single_doe_spectral(height_map=height_map, 
                                        aperture=self.aperture, 
                                        wavelengths=self.wavelengths,
                                        source_distance=self.source_distance,
                                        sensor_distance=self.sensor_distance,
                                        pixel_size=self.pixel_size,
                                        refractive_index=self.refractive_index,
                                        approximation=self.approximation)
    
    
    def forward_convolution(self, x, height_map):
        r"""
        Forward operator of the SingleDOESpectral layer.

        Args:
            x (torch.Tensor): Input tensor with shape (B, L, M, N)
        Returns:
            torch.Tensor: Output tensor with shape (B, 1, M, N) 
        """

        psf = self.get_psf(height_map)
        field = convolutional_sensing(x, psf, domain=self.domain)
        return self.sensor_spectral_sensitivity(field)


    def deconvolution(self, x, height_map, alpha=1e-3):
        r"""
        Backward operator of the SingleDOESpectral layer.

        Args:
            x (torch.Tensor): Input tensor with shape (B, 1, M, N)
        Returns:
            torch.Tensor: Output tensor with shape (B, L, M, N) 
        """

        psf = self.get_psf(height_map)
        x = self.sensor_spectral_sensitivity(x)
        return wiener_filter(x, psf, alpha)
    

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
