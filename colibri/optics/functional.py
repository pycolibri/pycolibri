import torch
import numpy as np

def prism_operator(x:torch.Tensor, shift_sign:int = 1) -> torch.Tensor:
    r"""

    Prism operator, shifts linearly the input tensor x in the spectral dimension.

    Args:
        x (torch.Tensor): Input tensor with shape (B, L, M, N)
        shift_sign (int): Integer, it can be 1 or -1, it indicates the direction of the shift
            if 1 the shift is to the right, if -1 the shift is to the left
    Returns:
        torch.Tensor: Output tensor with shape (1, L, M, N+L-1) if shift_sign is 1, or (1, L, M, N-L+1) if shift_sign is -1

    """

    assert shift_sign == 1 or shift_sign == -1, "The shift sign must be 1 or -1"
    _, L, M, N = x.shape  # Extract spectral image shape

    x = torch.unbind(x, dim=1)

    if shift_sign == 1:
        # Shifting produced by the prism 
        x = [torch.nn.functional.pad(x[l], (l, L - l - 1)) for l in range(L)]
    else:
        # Unshifting produced by the prism
        x = [x[l][:, :, l:N - (L - 1) + l] for l in range(L)]

    x = torch.stack(x, dim=1)
    return x

def forward_color_cassi(x: torch.Tensor, ca: torch.Tensor)-> torch.Tensor:

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
    y = prism_operator(y, shift_sign=1)
    return y.sum(dim=1, keepdim=True)

def backward_color_cassi(y: torch.Tensor, ca: torch.Tensor)-> torch.Tensor:
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
    y = prism_operator(y, shift_sign=-1)
    x = torch.multiply(y, ca)
    return x


def forward_dd_cassi(x: torch.Tensor, ca: torch.Tensor)-> torch.Tensor:
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
    ca = prism_operator(ca, shift_sign=-1)
    y = torch.multiply(x, ca)
    return y.sum(dim=1, keepdim=True)


def backward_dd_cassi(y: torch.Tensor, ca: torch.Tensor)-> torch.Tensor:
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
    ca = prism_operator(ca, shift_sign=-1)
    return torch.multiply(y, ca)

def forward_sd_cassi(x: torch.Tensor, ca: torch.Tensor)-> torch.Tensor:
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
    y2 = prism_operator(y1, shift_sign=1)
    return y2.sum(dim=1, keepdim=True)


def backward_sd_cassi(y: torch.Tensor, ca: torch.Tensor) -> torch.Tensor:
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
    y = prism_operator(y, shift_sign=-1)
    return torch.multiply(y, ca)


def forward_spc(x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
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
    x = x.contiguous().view(B, L, M * N)
    x = x.permute(0, 2, 1)

    # measurement
    H = H.unsqueeze(0).repeat(B, 1, 1)
    y = torch.bmm(H, x)
    return y


def backward_spc(y: torch.Tensor, H: torch.Tensor, pinv=False) -> torch.Tensor:
    r"""

    Inverse operation to reconstruct the image from measurements.

    For more information refer to: Optimized Sensing Matrix for Single Pixel Multi-Resolution Compressive Spectral Imaging  10.1109/TIP.2020.2971150

    Args:
        y (torch.Tensor): Measurement tensor of size (B, S, L).
        H (torch.Tensor): Measurement matrix of size (S, M*N).
        pinv (bool): Boolean, if True the pseudo-inverse of H is used, otherwise the transpose of H is used, defaults to False.
    Returns:
        torch.Tensor: Reconstructed image tensor of size (B, L, M, N).
    """

    Hinv   = torch.pinverse(H) if pinv else torch.transpose(H, 0, 1)
    Hinv   = Hinv.unsqueeze(0).repeat(y.shape[0], 1, 1)

    x = torch.bmm(Hinv, y)
    x = x.permute(0, 2, 1)
    b, c, hw = x.size()
    h = int(np.sqrt(hw))
    x = x.reshape(b, c, h, h)
    return x


### Wave optics

def get_spatial_coords(M: int, N: int, pixel_size: float, device=torch.device('cpu'), type='cartesian') -> tuple:
    r"""
    
    Generate the spatial coordinates for wave optics simulations in a specific coordinate system.
    
    .. note::
        * if type is 'cartesian', we generate :math:`(x, y)` coordinates in the Cartesian coordinate system, where
            * :math:`x \in \bigg[-\frac{\Delta\cdot N}{2}, \frac{\Delta\cdot N}{2} \bigg]`
            * :math:`y \in \bigg[-\frac{\Delta\cdot M}{2}, \frac{\Delta\cdot M}{2} \bigg]`
        * if type is 'polar', we generate :math:`(r, \theta)` coordinates in the Polar coordinate system, where
            * :math:`r \in \Bigg[0, \sqrt{\bigg(\frac{\Delta\cdot N}{2}\bigg)^2 + \bigg(\frac{\Delta\cdot M}{2}\bigg)^2} \Bigg]`
            * :math:`\theta \in [-\pi, \pi]`
        
        with :math:`\Delta` being the pixel size, :math:`M` the number of pixels in the y axis, and :math:`N` the number of pixels in the x axis.

    Args:
        M (int): number of pixels in the y axis.
        N (int): number of pixels in the x axis.
        pixel_size (float): Pixel size in meters.
        device (torch.device): Device, for more see torch.device().
        type (str): Type of coordinate system to generate ('cartesian' or 'polar').
    
    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple of tensors representing the X and Y coordinates
                                            if 'cartesian', or radius (r) and angle (theta) if 'polar'.


    """


    x = (torch.linspace(-pixel_size*N/2, pixel_size*N/2, N)).to(device=device)
    y = (torch.linspace(-pixel_size*M/2, pixel_size*M/2, M)).to(device=device)
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
    
    where :math:`\lambda` is the wavelength in meters.

    Args:
        wavelength (torch.Tensor): Wavelength in meters.
    Returns:
        torch.Tensor: Wavenumber.
    """
    return 2 * torch.pi / wavelength


def transfer_function_fresnel(M: int, 
                            N: int, 
                            pixel_size: float, 
                            wavelengths: torch.Tensor,
                            distance: float, 
                            device=torch.device('cpu')) -> torch.Tensor:
    r"""

    The transfer function for the Fresnel propagation can be written as follows:

    .. math::
        H(f_x, f_y, \lambda) = e^{j k s \left(1 - \frac{\lambda^2}{2} (f_x^2 + f_y^2)\right)}

    where :math:`f_x` and :math:`f_y` are the spatial frequencies, :math:`\lambda` is the wavelength, :math:`s` is the distance of propagation and :math:`k` is the wavenumber.


    Args:
        M (int): Resolution at Y axis in pixels.
        N (int): Resolution at N axis in pixels.
        pixel_size (float): Pixel pixel_size in meters.
        wavelengths (torch.Tensor): Wavelengths in meters.
        distance (float): Distance in meters.
        device (torch.device): Device, for more see torch.device().
    Returns:
        torch.Tensor: Complex kernel in Fourier domain with shape (len(wavelengths), M, N).
    """
    fr,_ = get_spatial_coords(M, N, 1/(N*pixel_size), device=device, type='polar')
    fr = fr.unsqueeze(0)
    H = torch.exp(1j * wave_number(wavelengths) * distance * (1 - ((fr**2) * (wavelengths**2)/2)) )
    return H


def transfer_function_angular_spectrum(M: int, N: int, pixel_size: float, wavelengths: torch.Tensor, distance: float, device: torch.device=torch.device('cpu')) -> torch.Tensor:
    r"""

    The transfer function for the angular spectrum propagation can be written as follows:

    .. math::
        H(f_x, f_y, \lambda) = e^{\frac{j s 2 \pi}{\lambda} \sqrt{1 - \lambda^2 (f_x^2 + f_y^2)}}

    where :math:`f_x` and :math:`f_y` are the spatial frequencies, :math:`\lambda` is the wavelength, :math:`s` is the distance of propagation and :math:`k` is the wavenumber.

    Args:
        M (int): Resolution at Y axis in pixels.
        N (int): Resolution at X axis in pixels.
        pixel_size (float): Pixel pixel_size in meters.
        wavelengths (torch.Tensor): Wavelengths in meters.
        distance (float): Distance in meters.
        device (torch.device): Device, for more see torch.device().
        type (str): Type of coordinates, can be "cartesian" or "polar".
    Returns:
        torch.Tensor: Complex kernel in Fourier domain with shape (len(wavelengths), M, N).
    """

    fr,_ = get_spatial_coords(M, N, 1/(N*pixel_size), device=device, type='polar')
    fr = fr.unsqueeze(0)
    H = torch.exp(1j * wave_number(wavelengths) * distance * (1 - ((fr**2) * wavelengths**2)) ** 0.5)

    return H


def fraunhofer_propagation(field: torch.Tensor, M: int, N: int, pixel_size: float, wavelengths: torch.Tensor, distance: float, device: torch.device=torch.device('cpu'))-> torch.Tensor:
    r"""
    The Fraunhofer approximation of :math:`U_0(x',y')` is its Fourier transform, :math:`\mathcal{F}\{U_0\}` with an additional phase factor that depends on the distance of propagation, :math:`z`. The Fraunhofer approximation is given by the following equation:

    .. math::
        U(x,y,z) \approx \frac{e^{jkz}e^{\frac{jk\left(x^2+y^2\right)}{2z}}}{j\lambda z} \mathcal{F}\left\{U_0(x,y)\right\}\left(\frac{x}{\lambda z}, \frac{y}{\lambda z}\right)

    where :math:`U(x,y,z)` is the field at distance :math:`z` from the source, :math:`U_0(x,y)` is the field at the source, :math:`\mathcal{F}` is the Fourier transform operator, :math:`k` is the wavenumber, :math:`\lambda` is the wavelength, :math:`\frac{x}{\lambda z}` and  :math:`\frac{y}{\lambda z}` are the spatial frequencies, and :math:`z` is the distance of propagation.

    Args:
        field (torch.Tensor): Input field.
        M (int): Resolution at Y axis in pixels.
        N (int): Resolution at X axis in pixels.
        pixel_size (float): Pixel pixel_size in meters.
        wavelengths (torch.Tensor): Wavelengths in meters.
        distance (float): Distance in meters.
        device (torch.device): Device, for more see torch.device().

    Returns:
        torch.Tensor: Propagated field. 
    """
    r, _ = get_spatial_coords(M, N, pixel_size, device=device, type='polar')
    r = r.unsqueeze(0)
    c = torch.exp(1j * wave_number(wavelengths) * distance) / (1j * wavelengths * distance) * torch.exp(1j * wave_number(wavelengths) / (2 * distance) * (r**2))
    c = c.to(device=device)
    result =  fft(field) * c * pixel_size**2

    return result


def fraunhofer_inverse_propagation(field: torch.Tensor, pixel_size: float, wavelengths: torch.Tensor, distance: float, device: torch.device=torch.device('cpu')) -> torch.Tensor:
    r"""
    The inverse Fraunhofer approximation (to reconstruct the field at the source from the field at the sensor) is given by the following equation:

    .. math::
        U_0(x,y) \approx \frac{1}{j\lambda z} e^{j k z} e^{\frac{j k (x^2 + y^2)}{2z}} \mathcal{F}^{-1}\left\{ U(x,y) \right\}

    where :math:`U_0(x,y)` is the field at the source, :math:`U(x,y)` is the field at the sensor, :math:`\mathcal{F}^{-1}` is the inverse Fourier transform operator,  :math:`k` is the wavenumber, :math:`\lambda` is the wavelength, and :math:`z` is the distance of propagation.   

    Args:
        field (torch.Tensor): Field at the sensor.
        pixel_size (float): Pixel pixel_size in meters.
        wavelengths (torch.Tensor): Wavelengths in meters.
        distance (float): Distance in meters.
        device (torch.device): Device, for more see torch.device().
    
    Returns:
        torch.Tensor: Reconstructed field.
    
    """
    _, M, N = field.shape
    r, _ = get_spatial_coords(M, N, pixel_size, device=device, type='polar')
    r = r.unsqueeze(0)
    c = torch.exp(1j * wave_number(wavelengths) * distance) / (1j * wavelengths * distance) * torch.exp(1j * wave_number(wavelengths) / (2 * distance) * r**2)
    c = c.to(device=device)
    result =  ifft(field / pixel_size**2 / c)
    
    return result


def fft(field: torch.Tensor, axis = (-2, -1)) -> torch.Tensor:
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


def ifft(field: torch.Tensor, axis = (-2, -1)) -> torch.Tensor:
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


def scalar_diffraction_propagation(field: torch.Tensor, distance: float, pixel_size: float, wavelength: list, approximation: str) -> torch.Tensor:
    r"""
    Compute the optical field propagation using a scalar diffraction theory model which is given by the specific approximation selected.
    
    
    .. note::
        * if 'approximation' is 'fresnel', the transfer function is calculated using :func:`colibri.optics.functional.transfer_function_fresnel`.
        * if 'approximation' is 'angular_spectrum', the transfer function is calculated using :func:`colibri.optics.functional.transfer_function_angular_spectrum`.
        * if 'approximation' is 'fraunhofer', the transfer function is calculated using :func:`colibri.optics.functional.fraunhofer_propagation`.
        * if 'approximation' is 'fraunhofer_inverse', the transfer function is calculated using :func:`colibri.optics.functional.fraunhofer_inverse_propagation`.

    Args:
        field (torch.Tensor): Input optical field of shape (C, M, N).
        distance (float): Propagation distance in meters.
        pixel_size (float): Pixel size in meters.
        wavelength (list): List of wavelengths in meters.
        approximation (str): Approximation (or diffraction model type) to use, can be "fresnel", "angular_spectrum" or "fraunhofer".
    Returns:
        torch.Tensor: The propagated optical field according to the selected approximation of shape (C, M, N).
    """

    _, M, N = field.shape
    
    if approximation == "fresnel":
        
        H = transfer_function_fresnel(M, 
                                    N, 
                                    pixel_size, 
                                    wavelength, 
                                    distance, 
                                    field.device)
    elif approximation == "angular_spectrum":
        H = transfer_function_angular_spectrum(M, 
                                            N, 
                                            pixel_size, 
                                            wavelength, 
                                            distance, 
                                            field.device)
        
    elif approximation == "fraunhofer":
        return fraunhofer_propagation(field, 
                                        M, 
                                        N, 
                                        pixel_size, 
                                        wavelength, 
                                        distance, 
                                        field.device)

    elif approximation == "fraunhofer_inverse":
        return fraunhofer_inverse_propagation(field,
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


def circular_aperture(M: int, N: int, radius: float, pixel_size: float) -> torch.Tensor:
    r'''
    Create a circular aperture mask of a given radius and pixel_size of size (M, N).

    .. math::
        A(x, y) = \begin{cases} 
            1 & \text{if } \pm \sqrt{(x^2 + y^2)} \leq \text{radius} \\
            0 & \text{otherwise}
        \end{cases}
        
    where: :math:`(x, y)` are the coordinates of each pixel, normalized by the pixel size, :math:`\text{radius}` is the radius of the aperture
    
    Args:
        M (int): Resolution at Y axis in pixels.
        N (int): Resolution at X axis in pixels.
        radius (float): Radius of the circular aperture in meters.
        pixel_size (float): Pixel size in meters.
    
    Returns:
        torch.Tensor: A binary mask with 1's inside the radius and 0's outside.
    '''
    r, _ = get_spatial_coords(M, N, pixel_size, type='polar')
    return r<=radius


def height2phase(height_map: torch.Tensor, wavelengths: torch.Tensor, refractive_index: callable) -> torch.Tensor:
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
                        sensor_distance:float, pixel_size: float, approximation:str = "fresnel") -> torch.Tensor:
    r"""
    Calculate the point spread function (PSF) of an optical system comprising a diffractive optical element (DOE) for spectral imaging. The PSF is calculated as follows:
    
    .. math::
        \mathbf{H}(\learnedOptics)  = |\mathcal{P_2}(z_2, \lambda) \left( \mathcal{P_1}(z_1,  \lambda)(\delta) * \learnedOptics \right)|^2
    
    where :math:`\mathcal{P_1}` is an operator that describes the propagation of light from the source to the DOE at a distance :math:`z_1`, :math:`\mathcal{P_2}` is an operator that describes the propagation of light from the DOE to the sensor at a distance :math:`z_2`, and :math:`\learnedOptics` is the DOE. 
    
    The operator :math:`\mathcal{P_2}` depends on the given approximation:

        - Fresnel :func:`colibri.optics.functional.transfer_function_fresnel`
        - Angular Spectrum :func:`colibri.optics.functional.transfer_function_angular_spectrum`
        - Fraunhofer :func:`colibri.optics.functional.fraunhofer_propagation`.
        
    
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
    M, N = height_map.shape
    wavelengths = wavelengths.unsqueeze(1).unsqueeze(2)
    k0 = wave_number(wavelengths)
    doe = height2phase(height_map = torch.unsqueeze(height_map, 0), wavelengths = wavelengths, refractive_index = refractive_index)
    doe = torch.exp(1j * doe*aperture)*aperture
    optical_field = torch.ones_like(doe)
    if not(np.isinf(source_distance) or np.isnan(source_distance)):
        r, _ = get_spatial_coords(M, N, pixel_size, device=doe.device, type='polar')
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


def gaussian_noise(y: torch.Tensor, snr: float) -> torch.Tensor:
    r"""
    Add Gaussian noise to an image based on a specified signal-to-noise ratio (SNR).

    .. math::
        \mathbf{\tilde{y}} = \mathbf{y} + n

    where :math:`n` is a Gaussian noise with zero mean and variance :math:`\sigma^2` such that is derived from the SNR and the power of :math:`\mathbf{y}`.

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



def fourier_conv(image: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
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

def add_pad(x: torch.Tensor, pad: list) -> torch.Tensor:
    r"""

    Add zero padding to a tensor.

    .. note::
        * pad object is a list of the same length as the number of dimensions of the tensor x
        * each element of the pad list is a integer, specifying the amount of padding to add on each side of the corresponding dimension in x.

    
    Args:
        x (torch.Tensor): Tensor to pad
        pad list:  padding to ad

    Returns:
        x (torch.Tensor): Padded tensor
        


    Example:
        >>> x = torch.tensor([[1, 2], [3, 4]])
        >>> add_pad(x, [1, 1])
        tensor([[0, 0, 0, 0],
                [0, 1, 2, 0],
                [0, 3, 4, 0],
                [0, 0, 0, 0]])

    """
    assert len(x.shape) == len(pad), "The tensor and the padding must have the same number of dimensions"

    pad_list = sum([[pa, pa] for pa in pad][::-1], [])
    return torch.nn.functional.pad(x, pad_list, mode='constant', value=0)

def unpad(x: torch.Tensor, pad: list) -> torch.Tensor:
    r"""

    Unpad a tensor.


    .. note::
        * pad is a list of the same length as the number of dimensions of the tensor x
        * each element of the pad list is a integer, specifying the amount of padding to remove on each side of the corresponding dimension in x.

    Args:
        x (torch.Tensor): Tensor to unpad
        pad int:  padding to remove
    Returns:
        x (torch.Tensor): Unpadded tensor


    Example:
        >>> x = torch.tensor([[0, 0, 0, 0],[0, 1, 2, 0],[0, 3, 4, 0],[0, 0, 0, 0]])
        >>> unpad(x, [1, 1])
        tensor([[1, 2],
                [3, 4]])

    """
    assert len(x.shape) == len(pad), "The tensor and the padding must have the same number of dimensions"
    if len(x.shape) ==4:
        return x[(0+pad[0]):(x.shape[0] - pad[0]), (0+pad[1]):(x.shape[1] - pad[1]), (0+pad[2]):(x.shape[2] - pad[2]), (0+pad[3]):(x.shape[3] - pad[3])]
    elif len(x.shape) ==3:
        return x[(0+pad[0]):(x.shape[0] - pad[0]), (0+pad[1]):(x.shape[1] - pad[1]), (0+pad[2]):(x.shape[2] - pad[2])]
    elif len(x.shape) ==2:
        return x[(0+pad[0]):(x.shape[0] - pad[0]), (0+pad[1]):(x.shape[1] - pad[1])]
    else:
        raise ValueError("The tensor must have 2, 3 or 4 dimensions")
    

def signal_conv(image: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
    r"""
    This function applies the convolution of an image with a Point Spread Function (PSF) in the signal domain.

    .. math::
        g(x, y) = f(x, y) * h(x, y)

    where :math:`f(x, y)` is the input image, :math:`h(x, y)` is the PSF, and :math:`g(x, y)` is the convolved output.


    Args:
        image (torch.Tensor): Image to simulate the sensing (B, L, M, N)
        psf (torch.Tensor): Point Spread Function (1, L, M, N)

    Returns:
        torch.Tensor: Measurement (B, 1, M, N)

    """
    original_size = image.shape[-2:]
    psf = psf.unsqueeze(1)
    C, _, _, N = psf.shape
    image = torch.nn.functional.conv2d(image, torch.flip(psf, [-2, -1]), padding=(N-1, N-1), groups=C)

    new_size = image.shape[-2:]
    image = unpad(image, pad = [0, 0, (new_size[0]-original_size[0])//2, (new_size[1]-original_size[1])//2])
    return image

def convolutional_sensing(image: torch.Tensor, psf: torch.Tensor, domain='fourier') -> torch.Tensor:
    r"""
    Simulate the convolutional sensing model of an optical system, using either Fourier or spatial domain methods.

    The "domain" argument choose to perform the convolution in the Fourier domain with :func:`colibri.optics.functional.fourier_conv` or the spatial domain with :func:`colibri.optics.functional.signal_conv`.

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


def wiener_filter(image: torch.Tensor, psf: torch.Tensor, alpha: float) -> torch.Tensor:
    r"""

    This function applies the Wiener filter to an image.

    .. math::
        \begin{aligned}
            X(x, y) &= \mathcal{F}^{-1}\{Y(u, v) \frac{H^*(u, v)}{|H(u, v)|^2 + \alpha}\}
        \end{aligned}

    where :math:`H(u, v)` is the optical transfer function, :math:`Y(u, v)` is the Fourier transform of the image, :math:`\alpha` is the regularization parameter, and :math:`X(x, y)` is the filtered image.

    Args:
        image (torch.Tensor): Image to apply the Wiener filter (B, L, M, N)
        psf (torch.Tensor): Point Spread Function (1, L, M, N)
        alpha (float): Regularization parameter

    Returns:
        torch.Tensor: Filtered image (B, L, M, N)

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
    filter = torch.conj(otf) / (torch.abs(otf) ** 2 + alpha)
    img_fft = img_fft * filter
    img = torch.abs(ifft(img_fft))

    if not(extra_size[0] < 0 or extra_size[1] < 0):
        img = unpad(img, pad = [0, 0, extra_size[0]//2, extra_size[1]//2])
    return img


def ideal_panchromatic_sensor(image: torch.Tensor) -> torch.Tensor:
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




def modulo(x, t=1.0):
    r"""
    Modulo operation.

    .. math::
        x = x - t \lfloor \frac{x}{t} \rfloor

    Args:
        x (torch.Tensor): Input tensor.
        t (float): Modulo value.

    Returns:
        torch.Tensor: Modulo operation result.

    """
    return x - t * torch.floor(x / t)