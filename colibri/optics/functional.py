import torch
import numpy as np

def prism_operator(x, shift_sign = 1):
    r"""

    Prism operator, shifts linearly the input tensor x in the spectral dimension.

    Args:
        x (torch.Tensor): Input tensor with shape (B, L, M, N)
        shift_sign (int): Integer, it can be 1 or -1, it indicates the direction of the shift
            if 1 the shift is to the right, if -1 the shift is to the left
    Returns:
        torch.Tensor: Output tensor with shape (1, L, M, N + L - 1) if shift_sign is 1, or (1, L M, N-L+1) if shift_sign is -1

    """

    assert shift_sign == 1 or shift_sign == -1, "The shift sign must be 1 or -1"
    _, L, M, N = x.shape  # Extract spectral image shape

    x = torch.unbind(x, dim=1)


    if shift_sign == 1:
        # Shifting produced by the prism 
        x = [torch.nn.functional.pad(x[l], (l, L - l - 1)) for l in range(L)]
    else:
        # Unshifting produced by the prism
        x = [x[l][:, :, l:N - (L- 1)+l] for l in range(L)]

    x = torch.stack(x, dim=1)
    return x

def forward_color_cassi(x, ca):

    r"""

    Forward operator of color coded aperture snapshot spectral imager (Color-CASSI)

    For more information refer to: Colored Coded Aperture Design by Concentration of Measure in Compressive Spectral Imaging https://doi.org/10.1109/TIP.2014.2310125

    Args:
        x (torch.Tensor): Spectral image with shape (B, L, M, N)
        ca (torch.Tensor): Coded aperture with shape (1, L, M, N)
    
    Returns: 
        torch.Tensor: Measurement with shape (B, 1, M, N + L - 1)
    """
    y = torch.multiply(x, ca)
    y = prism_operator(y, shift_sign = 1)
    return y.sum(dim=1, keepdim=True)

def backward_color_cassi(y, ca):
    r"""

    Backward operator of color coded aperture snapshot spectral imager (Color-CASSI)
    
    For more information refer to: Colored Coded Aperture Design by Concentration of Measure in Compressive Spectral Imaging https://doi.org/10.1109/TIP.2014.2310125

    Args:
        y (torch.Tensor): Measurement with shape (B, 1, M, N + L - 1)
        ca (torch.Tensor): Coded aperture with shape (1, L, M, N)
    Returns:
        torch.Tensor: Spectral image with shape (B, L, M, N)
    """
    y = torch.tile(y, [1, ca.shape[1], 1, 1])
    y = prism_operator(y, shift_sign = -1)
    x = torch.multiply(y, ca)
    return x


def forward_dd_cassi(x, ca):
    r"""

    Forward operator of dual disperser coded aperture snapshot spectral imager (DD-CASSI)
    
    For more information refer to: Single-shot compressive spectral imaging with a dual-disperser architecture https://doi.org/10.1364/OE.15.014013

    Args:
        x (torch.Tensor): Spectral image with shape (B, L, M, N)
        ca (torch.Tensor): Coded aperture with shape (1, 1, M, N + L - 1)
    Returns:
        torch.Tensor: Measurement with shape (B, 1, M, N)
    """
    _, L, M, N = x.shape  # Extract spectral image shape
    assert ca.shape[-1] == N + L - 1, "The coded aperture must have the same size as a dispersed scene"
    ca = torch.tile(ca, [1, L, 1, 1])
    ca = prism_operator(ca, shift_sign = -1)
    y = torch.multiply(x, ca)
    return y.sum(dim=1, keepdim=True)


def backward_dd_cassi(y, ca):
    r"""

    Backward operator of dual disperser coded aperture snapshot spectral imager (DD-CASSI)
    
    For more information refer to: Single-shot compressive spectral imaging with a dual-disperser architecture https://doi.org/10.1364/OE.15.014013

    Args:
        y (torch.Tensor): Measurement with shape (B, 1, M, N)
        ca (torch.Tensor): Coded aperture with shape (1, 1, M, N + L - 1)
    Returns:
        torch.Tensor: Spectral image with shape (1, L, M, N)
    """
    _, _, M, N_hat = ca.shape  # Extract spectral image shape
    L = N_hat - M + 1  # Number of shifts
    y = torch.tile(y, [1, L, 1, 1])
    ca = torch.tile(ca, [1, L, 1, 1])
    ca = prism_operator(ca, shift_sign = -1)
    return torch.multiply(y, ca)

def forward_sd_cassi(x, ca):
    r"""
    Forward operator of single disperser coded aperture snapshot spectral imager (SD-CASSI)
    
    For more information refer to: Compressive Coded Aperture Spectral Imaging: An Introduction: https://doi.org/10.1109/MSP.2013.2278763

    Args:
        x (torch.Tensor): Spectral image with shape (B, L, M, N)
        ca (torch.Tensor): Coded aperture with shape (1, 1, M, N)
    Returns:
        torch.Tensor: Measurement with shape (B, 1, M, N + L - 1)
    """
    y1 = torch.multiply(x, ca)  # Multiplication of the scene by the coded aperture
    _, M, N, L = y1.shape  # Extract spectral image shape
    # shift and sum
    y2 = prism_operator(y1, shift_sign = 1)
    return y2.sum(dim=1, keepdim=True)


def backward_sd_cassi(y, ca):
    r"""

    Backward operator of single disperser coded aperture snapshot spectral imager (SD-CASSI)
    
    For more information refer to: Compressive Coded Aperture Spectral Imaging: An Introduction: https://doi.org/10.1109/MSP.2013.2278763

    Args:
        y (torch.Tensor): Measurement with shape (B, 1, M, N + L - 1)
        ca (torch.Tensor): Coded aperture with shape (1, 1, M, N)
    Returns:
        torch.Tensor: Spectral image with shape (B, L, M, N)
    """
    _, _, M, N = y.shape  # Extract spectral image shape
    L = N - M + 1  # Number of shifts
    y = torch.tile(y, [1, L, 1, 1])
    y = prism_operator(y, shift_sign = -1)
    return torch.multiply(y, ca)


def forward_spc(x, H):
    r"""

    Forward propagation through the Single Pixel Camera (SPC) model.

    For more information refer to: Optimized Sensing Matrix for Single Pixel Multi-Resolution Compressive Spectral Imaging 10.1109/TIP.2020.2971150

    Args:
        x (torch.Tensor): Input image tensor of size (B, L, M, N).
        H (torch.Tensor): Measurement matrix of size (S, M*N).

    Returns:
        torch.Tensor: Output measurement tensor of size (B, S, L).
    """
    B, L, M, N = x.size()
    x = x.contiguous().view(B, L, M*N)
    x = x.permute(0, 2, 1)

    # measurement
    H = H.unsqueeze(0).repeat(B, 1, 1)
    y = torch.bmm(H, x)
    return y

def backward_spc(y, H):
    r"""

    Inverse operation to reconsstruct the image from measurements.

    For more information refer to: Optimized Sensing Matrix for Single Pixel Multi-Resolution Compressive Spectral Imaging  10.1109/TIP.2020.2971150

    Args:
        y (torch.Tensor): Measurement tensor of size (B, S, L).
        H (torch.Tensor): Measurement matrix of size (S, M*N).
    Returns:
        torch.Tensor: Reconstructed image tensor of size (B, L, M, N).
    """

    Hinv = torch.pinverse(H)
    Hinv = Hinv.unsqueeze(0).repeat(y.shape[0], 1, 1)

    x = torch.bmm(Hinv, y)
    x = x.permute(0, 2, 1)
    b, c, hw = x.size()
    h = int(np.sqrt(hw))
    x = x.reshape(b, c, h, h)
    return x


### Wave optics

def get_space_coords(ny: int, nx: int, pixel_size: float, device=torch.device('cpu'), type='cartesian'):
    r"""

    Space coordinates used in wave optics propagations.
        
    Args:
        ny (int): Resolution at Y axis in pixels.
        nx (int): Resolution at X axis in pixels.
        pixel_size (float): Pixel pixel_size in meters.
        device (torch.device): Device, for more see torch.device().
        type (str): Type of coordinates, can be "cartesian" or "polar".
    
    Returns:
        torch.Tensor: Space coordinates. Shape (ny, nx).
    """


    x = (torch.linspace(-pixel_size*nx/2, pixel_size*nx/2, nx)).to(device=device)
    y = (torch.linspace(-pixel_size*ny/2, pixel_size*ny/2, ny)).to(device=device)
    x,y = torch.meshgrid(y, x, indexing='ij')
    if type=="polar":
        r = torch.sqrt(x**2 + y**2)
        theta = torch.atan2(y, x)
        return r, theta
    else:
        return x, y
    

def wave_number(wavelength: torch.Tensor):
    r"""
    Wavenumber of a wave.

    Args:
        wavelength (torch.Tensor): Wavelength in meters.

    Returns:
        torch.Tensor: Wavenumber.
    """
    return 2 * torch.pi / wavelength


def transfer_function_fresnel(nu: int, 
                            nv: int, 
                            pixel_size: float, 
                            wavelengths: torch.Tensor,
                            distance: float, 
                            device=torch.device('cpu')):
    r"""

    The transfer function for the Fresnel propagation can be written as follows:

    .. math::
        H(f_x, f_y, \lambda) = e^{j k s \sqrt{1 - \lambda^2 (f_x^2 + f_y^2)}}

    where :math:`f_x` and :math:`f_y` are the spatial frequencies, :math:`\lambda` is the wavelength, :math:`s` is the distance of propagation and :math:`k` is the wavenumber.


    Args:
        nu (int): Resolution at X axis in pixels.
        nv (int): Resolution at Y axis in pixels.
        pixel_size (float): Pixel pixel_size in meters.
        wavelengths (torch.Tensor): Wavelengths in meters.
        distance (float): Distance in meters.
        device (torch.device): Device, for more see torch.device().
    Returns:
        torch.Tensor: Complex kernel in Fourier domain with shape (len(wavelengths), nu, nv).
    """
    fr,_ = get_space_coords(nv, nu, 1/(nu*pixel_size), device=device, type='polar')
    fr = fr.unsqueeze(0)
    H = torch.exp(1j * wave_number(wavelengths) * distance * (1 - ((fr**2) * wavelengths**2)) ** 0.5)
    return H

#     fx = torch.linspace(-1. / 2. / pixel_size, 1. / 2. / pixel_size, nu, dtype=torch.float32, device=device)
    # fy = torch.linspace(-1. / 2. / pixel_size, 1. / 2. / pixel_size, nv, dtype=torch.float32, device=device)
    # FY, FX = torch.meshgrid(fx, fy, indexing='ij')
    # fr =FX ** 2 + FY ** 2
    # fr = fr.unsqueeze(0)
    # H = torch.exp(1j * wave_number(wavelengths) * distance * (1 - (fr * wavelengths**2)) ** 0.5)
    # return H


def fft(field: torch.Tensor, axis = (-2, -1)):
    r"""

    Fast Fourier Transform of an optical field

    Args:
        field (torch.Tensor): Input field.
        axis (tuple): Tuple with the axes to perform the fft.
    Returns:
        torch.Tensor: Fourier transform of the input field.
    """
    field = torch.fft.fftshift(field, dim=axis)
    field = torch.fft.fft2(field, dim=axis)
    field = torch.fft.fftshift(field, dim=axis)

    return field


def ifft(field: torch.Tensor, axis = (-2, -1)):
    r"""
    
    Inverse Fast Fourier Transform of an optical field

    Args:
        field (torch.Tensor): Input field.
        axis (tuple): Tuple with the axes to perform the ifft.
    Returns:
        torch.Tensor: Inverse Fourier transform of the input field.
    """

    field = torch.fft.ifftshift(field, dim=axis)
    field = torch.fft.ifft2(field, dim=axis)
    field = torch.fft.ifftshift(field, dim=axis)
    return field


def scalar_diffraction_propagation(field: torch.Tensor, distance: float, pixel_size: float, wavelength: list, approximation: str):
    r"""
    The optical field propagation using scalar diffraction theory is given by the following equation: 
    
    .. math::
        U_2(x, y) = \mathcal{F}^{-1}\left\{ \mathcal{F}\{U_1(x, y)\} H(f_x, f_y, \lambda) \right\} 
    
    where :math:`U_1(x, y)` is the input field, :math:`U_2(x, y)` is the output field, :math:`H(f_x, f_y, \lambda)` is the transfer function and :math:`\mathcal{F}` is the Fourier transform operator.
    
    For more information see Goodman, J. W. (2005). Introduction to Fourier optics. Roberts and Company Publishers.
    Args:
        field (torch.Tensor): Input field. Shape (len(wavelengths), nu, nv).
        distance (float): Distance in meters.
        pixel_size (float): Pixel pixel_size in meters.
        wavelength (list): List of wavelengths in meters.
        approximation (str): Approximation to use, can be "fresnel", "angular_spectrum" or "Fraunhofer".
    Returns:
        torch.Tensor: Output field. Shape (len(wavelengths), nu, nv).
    """

    _, nu, nv = field.shape
    if approximation == "fresnel":
        H = transfer_function_fresnel(nu, 
                                    nv, 
                                    pixel_size, 
                                    wavelength, 
                                    distance, 
                                    field.device)
    else:
        raise NotImplementedError(f"{approximation} approximation is implemented")
    
    
    U1 = fft(field)
    U2 = U1 * H
    result = ifft(U2)
    return result


def circular_aperture(ny: int, nx: int, radius: float, pixel_size: float):
    r'''

    Create a circular aperture mask of a given radius and pixel_size of size (ny, nx).
    
    Args:
        nx (int): Resolution at X axis in pixels.
        ny (int): Resolution at Y axis in pixels.
        radius (float): Radius of the aperture.
        pixel_size (float): Pixel pixel_size in meters.
    
    '''
    r, _ = get_space_coords(ny, nx, pixel_size, type='polar')
    return r<=radius


def height2phase(doe: torch.Tensor, wavelengths: torch.Tensor, refractive_index: callable):
    r"""

    Convert height map to phase modulation.

    .. math::

        \Phi_{\text{DOE}}(x,y,\lambda) = k (n(\lambda) - 1) h(x, y)
    
    where :math:`\Phi_{\text{DOE}}` is the phase modulation, :math:`k` is the wavenumber, :math:`n(\lambda)` is the refractive index, :math:`h(x, y)` is the height map.

    Args:
        doe (torch.Tensor): Height map.
        wavelengths (torch.Tensor): Wavelengths in meters.
        refractive_index (function): Function to calculate the refractive index.
    Returns:
    torch.Tensor: Phase.    
    """
    k0 = wave_number(wavelengths)
    phase_doe =  refractive_index(wavelengths) * k0 * doe
    return phase_doe


def psf_single_doe_spectral(height_map: torch.Tensor, aperture: torch.Tensor, refractive_index: callable,
                        wavelengths: torch.Tensor, source_distance: float, 
                        sensor_distance:float, pixel_size: float, approximation = "fresnel"):
    r"""
    This function calculates the point spread fucntion (PSF) of an optical system composed by a DOE for spectral imaging.

    .. math::
        \begin{aligned}
            U_1(x, y) &=  \frac{1}{j\lambda s} e^{j \frac{k}{2s}(x^2 + y^2)}\\
            t(x,y) &= e^{i \Phi_{\text{DOE}}(x,y,\lambda)}\\
            U_2(x, y) &= U_1(x, y) t(x, y) A(x, y)\\ 
            \text{PSF} &= U_{\text{FPA}} = |\mathcal{F}^{-1}\left\{ \mathcal{F}\{U_2(x, y)\} H(f_x, f_y, \lambda) \right\} |^2
        \end{aligned}

    where :math:`U_1(x, y)` is the electric field before the phase mask, represented by the paraxial approximation of a spherical wave, :math:`U_2(x, y)` is the electric field after the phase mask. :math:`t(x, y)` denotes the phase transformation, :math:`A(x, y)` is the amplitude aperture function, and :math:`U_{\text{FPA}}` is the electric field in front of the sensor.
    
    
    Args:
        height_map (torch.Tensor): Height map of the DOE.
        aperture (torch.Tensor): Aperture mask.
        refractive_index (callable): Function to calculate the refractive index.
        wavelengths (torch.Tensor): Wavelengths in meters.
        source_distance (float): Source distance in meters.
        sensor_distance (float): Sensor distance in meters.
        pixel_size (float): Pixel pixel_size in meters.
        approximation (str): Approximation to use, can be "fresnel", "angular_spectrum" or "Fraunhofer".
    Returns:
        torch.Tensor: PSF of the optical system.
    """
    
    height_map = height_map*aperture
    ny, nx = height_map.shape
    wavelengths = wavelengths.unsqueeze(1).unsqueeze(2)
    k0 = wave_number(wavelengths)
    doe = height2phase(doe = torch.unsqueeze(height_map, 0), wavelengths = wavelengths, refractive_index = refractive_index)
    doe = torch.exp(1j * doe*aperture)*aperture
    optical_field = torch.ones_like(doe)
    if not(np.isinf(source_distance) or np.isnan(source_distance)):
        r, _ = get_space_coords(ny, nx, pixel_size, device=doe.device, type='polar')
        spherical_phase = (k0/(2*source_distance))*(torch.unsqueeze(r, 0)**2)
        optical_field = torch.exp(1j*spherical_phase) * (1/(1j*wavelengths*source_distance))

    optical_field = optical_field * doe
    optical_field_in_sensor = scalar_diffraction_propagation(field = optical_field, 
                                                            distance = sensor_distance, 
                                                            pixel_size = pixel_size, 
                                                            wavelength = wavelengths, 
                                                            approximation = approximation)
    psf = torch.abs(optical_field_in_sensor)**2
    #psf = psf/torch.norm(psf, p=1, dim=(-2, -1), keepdim=True)
    psf = psf/torch.sum(psf, dim=(-2, -1), keepdim=True)
    return psf


def addGaussianNoise(y: torch.Tensor, snr: float):
    r"""

    This function adds gaussian noise to an image
    y_noisy = x + noise
    Args:
        y (torch.Tensor): Image to add noise (B, L, M, N)
        snr (float): Signal to Noise Ratio
    Returns:
        y (torch.Tensor): Noisy image (B, L, M, N)
    """
    noise = torch.zeros_like(y)
    sigma_per_channel = torch.sum(torch.pow(y, 2), dim=(2, 3), keepdim=True) / (torch.numel(y[0,0,...]) * 10 ** (snr / 10))
    noise = torch.randn_like(y) * torch.sqrt(sigma_per_channel)
    return y+noise


def convolutional_sensing(image: torch.Tensor, psf: torch.Tensor):
    r"""
    This function simulates the convolutional sensing model of an optical system.
    Args:
        image (torch.Tensor): Image to simulate the sensing (B, L, M, N)
        psf (torch.Tensor): Point Spread Function (1, L, M, N)
    Returns:
        torch.Tensor: Measurement (B, 1, M, N)

    """
    img_fft = fft(image)
    otf = fft(psf)
    img_fft = img_fft * otf
    img = torch.abs(ifft(img_fft))
    return img


def weiner_filter(image: torch.Tensor, psf: torch.Tensor, alpha: float):
    r"""

    This function applies the Weiner filter to an image.

    .. math::
        \begin{aligned}
            W(u, v) &= \frac{H^*(u, v)}{|H(u, v)|^2 + \alpha} F(u, v)\\
            F'(u, v) &= F(u, v) W(u, v)\\
            f'(x, y) &= \mathcal{F}^{-1}\{F'(u, v)\}
        \end{aligned}
    where :math:`H(u, v)` is the optical transfer function, :math:`F(u, v)` is the Fourier transform of the image, :math:`\alpha` is the regularization parameter, :math:`W(u, v)` is the Weiner filter and :math:`f'(x, y)` is the filtered image.


    Args:
        image (torch.Tensor): Image to apply the Weiner filter (B, L, M, N)
        psf (torch.Tensor): Point Spread Function (1, L, M, N)
        alpha (float): Regularization parameter
    Returns:
        torch.Tensor: Filtered image (B, L, M, N)
    """
    img_fft = fft(image)
    otf = fft(psf)
    filter = torch.conj(otf) / (torch.abs(otf) ** 2 + alpha)
    img_fft = img_fft * filter
    img = torch.abs(ifft(img_fft))
    return img


def ideal_panchromatic_sensor(image: torch.Tensor):
    r"""
    This function simulates the ideal panchromatic sensor model of an optical system.
    Args:
        image (torch.Tensor): Image to simulate the sensing (B, L, M, N)
    Returns:
        torch.Tensor: Measurement (B, 1, M, N)
    """
    return torch.sum(image, dim=1, keepdim=True)/image.shape[1]