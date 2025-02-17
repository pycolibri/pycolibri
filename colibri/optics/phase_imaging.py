import torch
from colibri.optics.functional import coded_phase_imaging_forward, coded_phase_imaging_backward
from .utils import BaseOpticsLayer

class CodedPhaseImaging(BaseOpticsLayer):
    r"""
    Coded Phase Imaging 

    This optical system allow for the capture of a single intensity image of an object, and the subsequent reconstruction of the phase of the object. The system is composed of a phase mask that modulates the phase of the incoming light, and a sensor that captures the intensity of the light. The phase mask is designed to encode the phase information of the object in the intensity image. The phase of the object can be reconstructed from the intensity image using a phase retrieval algorithm.
    
    Mathematically, this system can be described as follows: 

    .. math::
    
            \mathbf{y} = \vert \forwardLinear_{\learnedOptics}(\mathbf{x}) \vert^2+ \noise
    
    where :math:`\noise` is the sensor noise, :math:`\mathbf{x}\in\xset` is the input optical field, :math:`\mathbf{y}\in\yset` are the acquired signal, for CodedPhaseImaging, :math:`\xset = \mathbb{C}^{M \times N}` and :math:`\yset = \mathbb{R}^{M \times N}`, and :math:`\forwardLinear_{\learnedOptics}:\xset\rightarrow \yset` is the forward operator of the optical system, which is defined as:

    .. math::

        \begin{align*}
        \forwardLinear_{\learnedOptics}: \mathbf{x} &\mapsto \mathbf{y} \\
                        \mathbf{y} &= \mathbf{y} = \left| \mathcal{P}_{(z, \lambda)}(\mathbf{x} \odot \learnedOptics) \right|^2 + \noise
        \end{align*}

    with :math:`\learnedOptics \in \mathbb{C}^{M \times N}` being the phase mask, :math:`\mathcal{P}_{(z, \lambda)}(\cdot)` the forward operator of the optical system, :math:`\odot` the element-wise product, and :math:`\noise` the sensor noise.

    """
    def __init__(self, 
                input_shape: torch.Size, 
                phase_mask: torch.Tensor = None, 
                pixel_size: float = 1e-6,
                wavelength: torch.Tensor = torch.tensor([550])*1e-9, 
                sensor_distance: float = 50e-3, 
                approximation: str = "fresnel",
                trainable: bool = False,
                ):
        r"""
        Initializes the CodedPhaseImaging layer.

        Args:
            input_shape (torch.Size): The shape of the input tensor.
            phase_mask (torch.Tensor, optional): The phase mask tensor. Defaults to None.
            pixel_size (float, optional): The size of each pixel in meters. Defaults to 1e-6.
            wavelength (torch.Tensor, optional): The wavelength of the light in meters. Defaults to torch.tensor([550])*1e-9.
            sensor_distance (float, optional): The distance from the sensor to the object in meters. Defaults to 50e-3.
            approximation (str, optional): The type of approximation to use (e.g., "fresnel"). Defaults to "fresnel".
            trainable (bool, optional): Whether the phase mask is trainable. Defaults to False.

        """
        
        self.input_shape = input_shape
        self.pixel_size = pixel_size
        self.wavelength = wavelength
        self.sensor_distance = sensor_distance
        self.approximation = approximation
        self.trainable = trainable


        self.M, self.N = input_shape  # Extract spectral image shape

        if phase_mask is None:
            phase_mask = torch.randn(self.M, self.N, requires_grad=trainable)  # Random initialization
        # Add parameter CA in pytorch manner
        phase_mask = torch.nn.Parameter(phase_mask, requires_grad=self.trainable)

        super(CodedPhaseImaging, self).__init__(
                                            learnable_optics=phase_mask, 
                                            sensing=lambda x, phase_mask: coded_phase_imaging_forward(x, phase_mask, pixel_size, wavelength, sensor_distance, approximation), 
                                            backward=lambda y, phase_mask: coded_phase_imaging_backward(y, phase_mask, pixel_size, wavelength, sensor_distance, approximation),
                                        )

    def intensity(self, x):
        r"""
        Computes the intensity of the input tensor.

        Args:
            x (torch.Tensor): Input tensor with shape (B, M, N)
        Returns:
            torch.Tensor: Intensity of the input tensor with shape (B, M, N)
        """
        return torch.abs(x)**2

    def forward(self, x, type_calculation="forward", intensity=False):
        r"""
        Performs the forward or backward operator according to the type_calculation

        Args:
            x (torch.Tensor): Input tensor with shape (B, M, N)
            type_calculation (str): String, it can be "forward", "backward" or "forward_backward"
        Returns:
            torch.Tensor: Output tensor with shape (B, M, N) 
        Raises:
            ValueError: If type_calculation is not "forward", "backward" or "forward_backward"
        """
        y = super(CodedPhaseImaging, self).forward(x, type_calculation)
        if intensity:
            y = self.intensity(y)
        return y


class FilteredSpectralInitialization(torch.nn.Module):
    r"""
    Filtered Spectral Initialization

    This layer implements the Filtered Spectral Initialization method for estimating the optical field from coded measurements of a Coded Phase Imaging System.
    
    Mathematically, this initialization can be described as follows: 

    .... TBD ....

    """
    def __init__(self, max_iterations:int , p:float, input_shape:tuple):
        super(FilteredSpectralInitialization, self).__init__()
        self.iter = max_iterations
        self.p = p 
        self.shape = input_shape 

        self.M = torch.tensor(self.shape[-2] * self.shape[-1], dtype=torch.float32)
        self.R = torch.ceil(self.M / self.p).to(torch.float32)
        

    def get_ytr(self, y): #Y0
        S = y.shape
        ytr_f = torch.zeros_like(y)  
        y_flat = y.view(S[0], -1)
        B = torch.argsort(y_flat, dim=1, descending=True)
        top_R_indices = B[:, :int(self.R)]
        ytr_flat = torch.zeros_like(y_flat)
        batch_indices = torch.arange(S[0]).unsqueeze(1).to(y.device)
        ytr_flat[batch_indices, top_R_indices] = 1
        ytr_f = ytr_flat.view(S)
        return ytr_f

        
    def forward(self, inputs, masks):
        Z0 = torch.randn(1, 1, self.shape[-2], self.shape[-1], dtype=torch.float32)
        Z0 = Z0 / torch.norm(Z0, p='fro')
        Y = inputs.float()
        normest = torch.sqrt(torch.sum(Y) / Y.numel())
        Z = self.Z0.to(inputs.device)
        Ytr = self.get_ytr(Y).to(torch.complex64).to(inputs.device)
        div = self.M * self.R
        for _ in range(self.iter):
            Z = torch.div(backward_propagator(Ytr * forward_propagator(Z, masks), masks), div)
            Z = Z / torch.norm(Z, p='fro')
            Z = Z.unsqueeze(1)
        Z = Z * normest.to(torch.complex64)
        Z = torch.abs(Z)
        return Z
    