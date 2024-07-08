import torch
import numpy as np
from colibri.optics.functional import get_space_coords, circular_aperture

def nbk7_refractive_index(wavelength):
    r"""
    
    nbk refractive index at a given wavelength

    Args:
        wavelength: Wavelength in meters

    Returns:
        val: Refractive index - 1

    """
    wavelength_squared = (wavelength * 1e6)**2
    n_power2_minus_1 = (1.03961212 * wavelength_squared )/(wavelength_squared - 0.00600069867) + (0.231792344 * wavelength_squared) / (wavelength_squared - 0.0200179144) + (1.01046945 * wavelength_squared) / (wavelength_squared - 103.560653)
    n = np.sqrt(n_power2_minus_1 + 1)
    return n - 1

def spiral_refractive_index(wavelength):
    r"""

    Spiral refractive index at a given wavelength

    Args:
        wavelength: Wavelength in meters

    Returns:
        val: Refractive index - 1

    """
    wavelength = (wavelength * 1e6)
    IdLens = 1.5375+0.00829045*wavelength**(-2)-0.000211046*wavelength**(-4)
    val = IdLens-1
    return val


def spiral_doe(M: int, N: int, number_spirals: int, radius: float, focal: float, start_w = 450e-9, end_w = 650e-9):

    r"""

    Code to generate a spiral DOE with a given number of spirals, radius, focal length and wavelength range.

    For more information, please refer to the following paper:  (2019). Compact snapshot hyperspectral imaging with diffracted rotation.

    Args:
    
        M (int): Resolution at Y axis in pixels.
        N (int): Resolution at X axis in pixels.
        number_spirals (int): Number of spirals.
        radius (float): Radius of the doe.
        focal (float): Focal length of the doe.
        start_w (float): Initial design wavelength.
        end_w (float): Final design wavelength.

    Returns:

        torch.Tensor: Height map of the spiral DOE
        torch.Tensor: Aperture of the spiral DOE
    """
    pixel_size = (2*radius)/np.min([M, N]) 
    r, theta = get_space_coords(M = M, N = N, pixel_size = pixel_size, type='polar')
    aperture = circular_aperture(M = M, N = N, radius = radius, pixel_size = pixel_size)
    theta = torch.remainder(theta + torch.pi, 
                            (2 * torch.pi / number_spirals))
    lt = start_w + (end_w - start_w) * number_spirals * theta / 2 / torch.pi
    n = torch.true_divide((torch.sqrt(r**2 + focal**2) - focal), lt)  # Constructive interference
    n = torch.ceil(n+1e-6)
    height_map = (n * lt - (torch.sqrt(r**2 + focal**2) - focal)) / spiral_refractive_index(wavelength=lt)  # Heights
    return height_map * aperture, aperture



def conventional_lens(M: int, N: int, focal = None, radius = None):
    r"""

    Code to generate a conventional lens with a given focal length and radius following the equation 

    .. math::
        h(x, y) = \frac{-(x^2 + y^2)}{f}

    where :math:`r` is the distance from the center of the lens and :math:`f` is the focal length of the lens.

    For more information, please refer to:  
    Goodman, J. W. (2005). Introduction to Fourier optics. Roberts and Company Publishers.
    (2017). Design and fabrication of diffractive optical elements with MATLAB.

    
    Args:
        M: Number of pixels in the y direction,
        N: Number of pixels in the x direction,
        focal: Focal length of the lens
        wavelength: Wavelength of the light
        radius: Radius of the lens
    Returns:
        torch.Tensor: Height map of the conventional lens
        torch.Tensor: Aperture of the conventional lens
    """
    pixel_size = (2*radius)/np.min([M, N])
    r, _ = get_space_coords(M = M, N = N, pixel_size = pixel_size, type='polar')
    aperture = circular_aperture(M = M, N = N, radius = radius, pixel_size = pixel_size)
    height_map = -(r**2)/(focal)
    return height_map*aperture, aperture
