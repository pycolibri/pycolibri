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

    Inverse operation to reconstruct the image from measurements.

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
    
    Generate spatial coordinates for wave optics simulations in either Cartesian or polar format.
        
    Args:
        ny (int): Resolution at Y axis in pixels.
        nx (int): Resolution at X axis in pixels.
        pixel_size (float): Pixel pixel_size in meters.
        device (torch.device): Device, for more see torch.device().
        type (str): Type of coordinate system to generate ('cartesian' or 'polar').
    
    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple of tensors representing the X and Y coordinates
                                           if 'cartesian', or radius (r) and angle (theta) if 'polar'.
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
    Wavenumber 
    
    .. math::
        k = \frac{2 \pi}{\lambda}
    
    where :math:`\lambda` is the wavelength.
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
        H(f_x, f_y, \lambda) = e^{j k s \left(1 - \frac{\lambda^2}{2} (f_x^2 + f_y^2)\right)}

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
    H = torch.exp(1j * wave_number(wavelengths) * distance * (1 - ((fr**2) * (wavelengths**2)/2)) )
    return H


def transfer_function_angular_spectrum(nu: int, nv: int, pixel_size: float, wavelengths: torch.Tensor, distance: float, device: torch.device=torch.device('cpu'), type='cartesian'):
    r"""

    The transfer function for the angular spectrum propagation can be written as follows:

    .. math::
        H(f_x, f_y, \lambda) = e^{\frac{j s 2 \pi}{\lambda} \sqrt{1 - \lambda^2 (f_x^2 + f_y^2)}}

    where :math:`f_x` and :math:`f_y` are the spatial frequencies, :math:`\lambda` is the wavelength, :math:`s` is the distance of propagation and :math:`k` is the wavenumber.

    Args:
        nu (int): Resolution at X axis in pixels.
        nv (int): Resolution at Y axis in pixels.
        pixel_size (float): Pixel pixel_size in meters.
        wavelengths (torch.Tensor): Wavelengths in meters.
        distance (float): Distance in meters.
        device (torch.device): Device, for more see torch.device().
        type (str): Type of coordinates, can be "cartesian" or "polar".
    Returns:
        torch.Tensor: Complex kernel in Fourier domain with shape (len(wavelengths), nu, nv).
    """

    fr,_ = get_space_coords(nv, nu, 1/(nu*pixel_size), device=device, type='polar')
    fr = fr.unsqueeze(0)
    H = torch.exp(1j * wave_number(wavelengths) * distance * (1 - ((fr**2) * wavelengths**2)) ** 0.5)

    return H


def fraunhofer_propagation(field: torch.Tensor, nu: int, nv: int, pixel_size: float, wavelengths: torch.Tensor, distance: float, device: torch.device=torch.device('cpu'), type='cartesian'):

    r"""
    Simulate Fraunhofer diffraction (far-field) propagation of a wave field.

    .. math::
        U(x, y) = \mathcal{F}^{-1} {A(f_x, f_y) exp(-j * \pi * \lambda * z * (f_x^2 + f_y^2))}

    where: :math:`\mathcal{F}^{-1}` is the inverse Fourier transform, :math:`A(f_x, f_y)` is the Fourier transform of the aperture function, :math:`\lambda` is the wavelength, :math:`z` is the propagation distance, :math:`f_x` and :math:`f_y` are spatial frequencies.

    Args:
        field (torch.Tensor): The input optical field.
        nu (int): Number of pixels along the horizontal axis of the output image.
        nv (int): Number of pixels along the vertical axis of the output image.
        pixel_size (float): Pixel size in the output image, in meters.
        wavelengths (torch.Tensor): Wavelengths of the light, in meters.
        distance (float): Propagation distance, in meters.
        device (torch.device): Computation device (e.g., CPU or GPU).
        type (str): Coordinate system type used for calculations ('cartesian' or 'polar').

    Returns:
        torch.Tensor: The propagated wave field at the given distance.
    """
    r, _ = get_space_coords(nv, nu, pixel_size, device=device, type='polar')
    r = r.unsqueeze(0)
    c = torch.exp(1j * wave_number(wavelengths) * distance) / (1j * wavelengths * distance) * torch.exp(1j * wave_number(wavelengths) / (2 * distance) * (r**2))
    c = c.to(device=device)
    result =  fft(field) * c * pixel_size**2

    return result


# def fraunhofer_inverse_propagation(field: torch.Tensor, nu: int, nv: int, pixel_size: float, wavelengths: torch.Tensor, distance: float, device: torch.device=torch.device('cpu'), type='cartesian'):
#     r"""
#     [TO DOCUMMENT]
#     [TO Delete]
#     """

#     r, _ = get_space_coords(nv, nu, pixel_size, device=device, type='polar')
#     r = r.unsqueeze(0)
#     c = torch.exp(-1j * wave_number(wavelengths) * distance) * torch.exp(-1j * wave_number(wavelengths) / (2 * distance) * r**2)
#     c = c.to(device=device)
#     result =  ifft(field / pixel_size**2 / c)

#     return result


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
    Compute the optical field propagation using a scalar diffraction theory model which is given by the following equation: 
    
    .. math::
        U_2(x, y) = \mathcal{F}^{-1}\left\{ \mathcal{F}\{U_1(x, y)\} H(f_x, f_y, \lambda) \right\} 
    
    where :math:`U_1(x, y)` is the input field, :math:`U_2(x, y)` is the output field, :math:`H(f_x, f_y, \lambda)` is the transfer function and :math:`\mathcal{F}` is the Fourier transform operator.
    
    For more information see Goodman, J. W. (2005). Introduction to Fourier optics. Roberts and Company Publishers.

    Args:
        field (torch.Tensor): Input optical field. Shape (len(wavelengths), nu, nv).
        distance (float): Propagation distance in meters.
        pixel_size (float): Pixel size in meters.
        wavelength (list): List of wavelengths in meters.
        approximation (str): Approximation (or diffraction model type) to use, can be "fresnel", "angular_spectrum" or "fraunhofer".
    Returns:
        torch.Tensor: The propagated optical field according to the selected approximation. Shape (len(wavelengths), nu, nv).
    """

    _, nu, nv = field.shape
    print("#"*10, approximation)
    if approximation == "fresnel":
        
        H = transfer_function_fresnel(nu, 
                                    nv, 
                                    pixel_size, 
                                    wavelength, 
                                    distance, 
                                    field.device)
    elif approximation == "angular_spectrum":
        H = transfer_function_angular_spectrum(nu, 
                                            nv, 
                                            pixel_size, 
                                            wavelength, 
                                            distance, 
                                            field.device)
        
    elif approximation == "fraunhofer":
        return fraunhofer_propagation(field, 
                                      nu, 
                                      nv, 
                                      pixel_size, 
                                      wavelength, 
                                      distance, 
                                      field.device)

    # elif approximation == "fraunhofer_inverse":
    #     return fraunhofer_inverse_propagation(field,
    #                                         nu,
    #                                         nv,
    #                                         pixel_size,
    #                                         wavelength,
    #                                         distance,
    #                                         field.device)

    else:
        raise NotImplementedError(f"{approximation} approximation is implemented")
    
    
    U1 = fft(field)
    U2 = U1 * H
    result = ifft(U2)
    return result


def circular_aperture(ny: int, nx: int, radius: float, pixel_size: float):
    r'''
    Create a circular aperture mask of a given radius and pixel_size of size (ny, nx).

    .. math::
        \begin{cases} 
            1 & \text{if } sqrt(x^2 + y^2) \leq \text{radius} \\
            0 & \text{otherwise}
        \end{cases}
        
    where: :math:`(x, y)` are the coordinates of each pixel, normalized by the pixel size, :math:`\text{radius}` is the radius of the aperture
    
    Args:
        nx (int): Resolution at X axis in pixels.
        ny (int): Resolution at Y axis in pixels.
        radius (float): Radius of the circular aperture in meters.
        pixel_size (float): Pixel size in meters.
    
    Returns:
        torch.Tensor: A binary mask with 1's inside the radius and 0's outside.
    '''
    r, _ = get_space_coords(ny, nx, pixel_size, type='polar')
    return r<=radius


def height2phase(height_map: torch.Tensor, wavelengths: torch.Tensor, refractive_index: callable):
    r"""

    Convert height map to phase modulation.

    .. math::

        \Phi(x,y,\lambda) = k(\lambda) \Delta n(\lambda) h(x, y)
    
    where :math:`\Phi` is the phase modulation,  :math:`h(x, y)` is the height map of the optical element, :math:`k(\lambda)` is the wavenumber for \lambda wavelength and :math:`\Delta n(\lambda)` is the change of refractive index between propagation medium and material of the optical element .

    Args:
        height_map (torch.Tensor): Height map.
        wavelengths (torch.Tensor): Wavelengths in meters.
        refractive_index (function): Function to calculate the refractive index.
    Returns:
        torch.Tensor: Phase.    
    """
    k0 = wave_number(wavelengths)
    phase_doe =  refractive_index(wavelengths) * k0 * height_map
    return phase_doe


def psf_single_doe_spectral(height_map: torch.Tensor, aperture: torch.Tensor, refractive_index: callable,
                        wavelengths: torch.Tensor, source_distance: float, 
                        sensor_distance:float, pixel_size: float, approximation = "fresnel"):
    r"""
    Calculate the point spread function (PSF) of an optical system comprising a diffractive optical element (DOE) for spectral imaging.
    
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
        source_distance (float): Distance from the source to the DOE in meters.
        sensor_distance (float): Distance from the DOE to the sensor in meters.
        pixel_size (float): Pixel size in meters.
        approximation (str): Type of propagation model ('fresnel', 'angular_spectrum', 'fraunhofer').

    Returns:
        torch.Tensor: PSF of the optical system, normalized to unit energy.
    """
    
    height_map = height_map*aperture
    ny, nx = height_map.shape
    wavelengths = wavelengths.unsqueeze(1).unsqueeze(2)
    k0 = wave_number(wavelengths)
    doe = height2phase(height_map = torch.unsqueeze(height_map, 0), wavelengths = wavelengths, refractive_index = refractive_index)
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
    psf = psf/torch.sum(psf, dim=(-2, -1), keepdim=True)
    return psf


def addGaussianNoise(y: torch.Tensor, snr: float):
    r"""
    Add Gaussian noise to an image based on a specified signal-to-noise ratio (SNR).

    .. math::
    y_noisy = y + n
    .. math::
    n ~ N(0, \sigma^2)

    where :math:`\sigma^2` is derived from the SNR and the power of :math:`y`.

    Args:
        y (torch.Tensor): Original image tensor with shape (B, L, M, N).
        snr (float): Desired signal-to-noise ratio in decibels (dB).

    Returns:
        torch.Tensor: Noisy image tensor with the same shape as input.
    """
    noise = torch.zeros_like(y)
    sigma_per_channel = torch.sum(torch.pow(y, 2), dim=(2, 3), keepdim=True) / (torch.numel(y[0,0,...]) * 10 ** (snr / 10))
    noise = torch.randn_like(y) * torch.sqrt(sigma_per_channel)
    return y+noise



def fourier_conv(image: torch.Tensor, psf: torch.Tensor):
    r"""
    Apply Fourier convolution theorem to simulate the effect of a linear system characterized by a point spread function (PSF).

    .. math::
    g = \mathcal{F}^{-1}(\mathcal{F}(f) * \mathcal{F}(h))

    where :math:`f` is the input image, :math:`h` is the PSF, :math:`g` is the convolved output, :math:`\mathcal{F}` and :math:`\mathcal{F}^{-1}` denote the Fourier and inverse Fourier transforms.

    Args:
        image (torch.Tensor): Image to simulate the sensing (B, L, M, N)
        psf (torch.Tensor): Point Spread Function (1, L, M, N)
    Returns:
        torch.Tensor: Measurement (B, 1, M, N)

    """
    # Fix psf and image size
    psf_size = psf.shape[-2:]
    image_size = image.shape[-2:]
    extra_size = [(psf_size[i]-image_size[i]) for i in range(len(image_size))]
    if extra_size[0] < 0 or extra_size[1] < 0:
        psf = add_pad(psf, [0, -extra_size[0]//2, -extra_size[1]//2])
    else:
        image = add_pad(image, [0, 0, extra_size[0]//2, extra_size[1]//2])

    img_fft = fft(image)
    otf = fft(psf)
    img_fft = img_fft * otf
    img = torch.abs(ifft(img_fft))
    if not(extra_size[0] < 0 or extra_size[1] < 0):
        img = unpad(img, pad = [0, 0, extra_size[0]//2, extra_size[1]//2])
    return img

def add_pad(x, pad):
    r"""
    Args:
        x (torch.Tensor): Tensor to pad
        pad int:  padding to ad
    Returns:
        x (torch.Tensor): Padded tensor
    """
    assert len(x.shape) == len(pad), "The tensor and the padding must have the same number of dimensions"

    pad_list = sum([[pa, pa] for pa in pad][::-1], [])
    return torch.nn.functional.pad(x, pad_list, mode='constant', value=0)

def unpad(x, pad):
    r"""
    Args:
        x (torch.Tensor): Tensor to unpad
        pad int:  padding to remove
    Returns:
        x (torch.Tensor): Unpadded tensor
    """
    assert len(x.shape) == len(pad), "The tensor and the padding must have the same number of dimensions"
    if len(x.shape) ==4:
        return x[(0+pad[0]):(x.shape[0] - pad[0]), (0+pad[1]):(x.shape[1] - pad[1]), (0+pad[2]):(x.shape[2] - pad[2]), (0+pad[3]):(x.shape[3] - pad[3])]
    elif len(x.shape) ==3:
        return x[(0+pad[0]):(x.shape[0] - pad[0]), (0+pad[1]):(x.shape[1] - pad[1]), (0+pad[2]):(x.shape[2] - pad[2])]
    elif len(x.shape) ==2:
        return x[(0+pad[0]):(x.shape[0] - pad[0]), (0+pad[1]):(x.shape[1] - pad[1])]
    else:
        raise ValueError("The tensor must have 3 or 4 dimensions")
    

def signal_conv(image: torch.Tensor, psf: torch.Tensor):
    r"""
    This function applies the convolution of an image with a Point Spread Function (PSF).
    Args:
        image (torch.Tensor): Image to simulate the sensing (B, L, M, N)
        psf (torch.Tensor): Point Spread Function (1, L, M, N)
    Returns:
        torch.Tensor: Measurement (B, 1, M, N)

    """
    original_size = image.shape[-2:]
    psf = psf.unsqueeze(1)
    C, _, _, Nx = psf.shape
    image = torch.nn.functional.conv2d(image, torch.flip(psf, [-2, -1]), padding=(Nx-1, Nx-1), groups=C)

    new_size = image.shape[-2:]
    image = unpad(image, pad = [0, 0, (new_size[0]-original_size[0])//2, (new_size[1]-original_size[1])//2])
    return image

def convolutional_sensing(image: torch.Tensor, psf: torch.Tensor, domain='fourier'):
    r"""
    Simulate the convolutional sensing model of an optical system, using either Fourier or spatial domain methods.

    In the "signal" domain, the convolution operation is performed in the spatial domain using the torch.nn.functional.conv2d function.
    .. math::
        g(x, y) = f(x, y) * h(x, y)

    In the "fourier" domain, the convolution operation is performed in the Fourier domain using the Fourier convolution theorem.
    .. math::
        \mathcal{G}(f_x, f_y) = \mathcal{F}(f_x, f_y) * \mathcal{H}(f_x, f_y)

    where :math:`f(x, y)` is the input image, :math:`h(x, y)` is the PSF, :math:`*` denotes the convolution operation, :math:`\mathcal{F}` and :math:`\mathcal{H}` are the Fourier transforms of :math:`f` and :math:`h`, respectively, and :math:`\mathcal{G}` is the Fourier transform of the output image :math:`g`.

    Args:
        image (torch.Tensor): Image tensor to simulate the sensing (B, L, M, N).
        psf (torch.Tensor): Point Spread Function (PSF) tensor (1, L, M, N).
        domain (str): Domain for convolution operation, 'fourier' or 'signal'.

    Returns:
        torch.Tensor: Convolved image tensor as measurement (B, 1, M, N).

    Raises:
        NotImplementedError: If the specified domain is not supported.

    """
    if domain == 'fourier':
        return fourier_conv(image, psf)
    elif domain == 'signal':
        return signal_conv(image, psf)
    else:
        raise NotImplementedError(f"{domain} domain is not implemented")


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

    Simulate the response of an ideal panchromatic sensor by averaging the spectral channels.

    .. math::
    I = \frac{1}{L} \sum_{\lambda} I_{\lambda}

    where :math:`I_{\lambda}` is the intensity at each wavelength, and :math:`L` is the number of spectral channels.

    Args:
        image (torch.Tensor): Multispectral image tensor (B, L, M, N).

    Returns:
        torch.Tensor: Simulated sensor output as measurement (B, 1, M, N).

    """
    return torch.sum(image, dim=1, keepdim=True)/image.shape[1]