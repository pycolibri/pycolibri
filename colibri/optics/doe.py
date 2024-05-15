import torch


def get_transfer_function_fresnel_kernel(nu: int, 
                                         nv: int, 
                                         dx: float, 
                                         wavelengths: list,
                                         distance: float, 
                                         device=torch.device('cpu')):
    r"""

    Transfer function for Fresnel propagation.

    Args:
        nu (int): Resolution at X axis in pixels.
        nv (int): Resolution at Y axis in pixels.
        dx (float): Pixel pitch in meters.
        wavelengths (list): List of wavelengths in meters.
        distance (float): Distance in meters.
        device (torch.device): Device, for more see torch.device().
    Returns:
        torch.Tensor: Complex kernel in Fourier domain with shape (len(wavelengths), nu, nv).
    """

    distance = torch.tensor(distance, device=device).unsqueeze(-1).unsqueeze(-1)
    fx = torch.linspace(-1. / 2. / dx, 1. / 2. / dx, nu, dtype=torch.float32, device=device)
    fy = torch.linspace(-1. / 2. / dx, 1. / 2. / dx, nv, dtype=torch.float32, device=device)
    FY, FX = torch.meshgrid(fx, fy, indexing='ij')

    FX, FY = FX.unsqueeze(0), FY.unsqueeze(0)  # add a dimension for wavelengths

    wavelengths = torch.tensor(wavelengths, device=device).unsqueeze(-1).unsqueeze(-1)
    k = 2 * torch.pi / wavelengths

    H = torch.exp(1j * k * distance * (1 - (FX * wavelengths) ** 2 - (FY * wavelengths) ** 2) ** 0.5)

    return H

def fft(field, axis = (-2, -1)):
    field = torch.fft.fftshift(field, dim=axis)
    field = torch.fft.fft2(field, dim=axis)
    field = torch.fft.fftshift(field, dim=axis)

    return field
def ifft(field, axis = (-2, -1)):
    field = torch.fft.ifftshift(field, dim=axis)
    field = torch.fft.ifft2(field, dim=axis)
    field = torch.fft.ifftshift(field, dim=axis)
    return field

def transfer_function_fresnel(field: torch.Tensor, distance: float, dx: float, wavelength: list):
    r"""


    Convolution Fresnel approximation.

    Args:
        field (torch.Tensor): Input field. Shape (len(wavelengths), nu, nv).
        distance (float): Distance in meters.
        dx (float): Pixel pitch in meters.
        wavelength (list): List of wavelengths in meters.

    Returns:
        torch.Tensor: Output field. Shape (len(wavelengths), nu, nv).
    """

    _, nu, nv = field.shape
    H = get_transfer_function_fresnel_kernel(nu, 
                                             nv, 
                                             dx, 
                                             wavelength, 
                                             distance, 
                                             field.device)
    
    U1 = fft(field)
    U2 = U1 * H
    result = ifft(U2)
    return result


def circular_aperture(diameter:int):
    r'''


    Create a circular aperture mask
    
    '''

    u = torch.linspace(-diameter // 2, diameter // 2 - 1, diameter)
    v = torch.linspace(-diameter // 2, diameter // 2 - 1, diameter)
    [x, y] = torch.meshgrid(u, v, indexing='xy')
    radius_distance = torch.sqrt(x ** 2 + y ** 2)<=diameter//2

    return radius_distance.float()