import torch
import numpy as np

def prism_operator(x, shift_sign = 1):
    """
    Prism operator, it shifts the input tensor x according to spectral shift made by a prism
    Args:
        x (torch.Tensor): Input tensor with shape (1, M, N, L)
        shift_sign (int): Integer, it can be 1 or -1, it indicates the direction of the shift
    Returns:
        torch.Tensor: Output tensor with shape (1, M, N + L - 1, L) if shift_sign is 1, or (1, M, N-L+1, L) if shift_sign is -1
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
    """
    Forward operator of color coded aperture snapshot spectral imager (Color-CASSI), more information refer to: Computational snapshot multispectral cameras: Toward dynamic capture of the spectral world https://doi.org/10.1109/MSP.2016.2582378
    Args:
        x (torch.Tensor): Spectral image with shape (1, M, N, L)
        ca (torch.Tensor): Coded aperture with shape (1, M, N, L)
    
    Returns: 
        torch.Tensor: Measurement with shape (1, M, N, 1)
    """
    y = torch.multiply(x, ca)
    return y.sum(dim=-1, keepdim=True)

def backward_color_cassi(y, ca):
    """
    Backward operator of color coded aperture snapshot spectral imager (Color-CASSI), more information refer to: Computational snapshot multispectral cameras: Toward dynamic capture of the spectral world https://doi.org/10.1109/MSP.2016.2582378
    Args:
        y (torch.Tensor): Measurement with shape (1, M, N, L)
        ca (torch.Tensor): Coded aperture with shape (1, M, N, L)
    Returns:
        torch.Tensor: Spectral image with shape (1, M, N, L)
    """
    x = torch.multiply(y, ca)
    return x


def forward_dd_cassi(x, ca):
    """
    Forward operator of dual disperser coded aperture snapshot spectral imager (DD-CASSI), more information refer to: Computational snapshot multispectral cameras: Toward dynamic capture of the spectral world https://doi.org/10.1109/MSP.2016.2582378
    Args:
        x (torch.Tensor): Spectral image with shape (1, M, N, L)
        ca (torch.Tensor): Coded aperture with shape (1, M, N + L - 1, 1)
    Returns:
        torch.Tensor: Measurement with shape (1, M, N + L - 1, 1)
    """
    _, L, M, N = x.shape  # Extract spectral image shape
    assert ca.shape[-2] == N + L - 1, "The coded aperture must have the same size as a dispersed scene"
    ca = torch.tile(ca, [1, L, 1, 1])
    ca = prism_operator(ca, shift_sign = -1)
    y = forward_color_cassi(x, ca)
    return y


def backward_dd_cassi(y, ca):
    """
    Backward operator of dual disperser coded aperture snapshot spectral imager (DD-CASSI), more information refer to: Computational snapshot multispectral cameras: Toward dynamic capture of the spectral world https://doi.org/10.1109/MSP.2016.2582378
    Args:
        y (torch.Tensor): Measurement with shape (1, M, N + L - 1, 1)
        ca (torch.Tensor): Coded aperture with shape (1, M, N + L - 1, 1)
    Returns:
        torch.Tensor: Spectral image with shape (1, M, N, L)
    """
    _, M, N, _ = ca.shape  # Extract spectral image shape
    L = N - M + 1  # Number of shifts
    y = torch.tile(y, [1, 1, 1, L])
    ca = torch.tile(ca, [1, 1, 1, L])
    ca = prism_operator(ca, shift_sign = -1)
    return backward_color_cassi(y, ca)

def forward_cassi(x, ca):
    """
    Forward operator of coded aperture snapshot spectral imager (CASSI), more information refer to: Compressive Coded Aperture Spectral Imaging: An Introduction: https://doi.org/10.1109/MSP.2013.2278763
    Args:
        x (torch.Tensor): Spectral image with shape (1, L, M, N,)
        ca (torch.Tensor): Coded aperture with shape (1, 1, M, N)
    Returns:
        torch.Tensor: Measurement with shape (1, 1, M, N + L - 1)
    """
    y1 = torch.multiply(x, ca)  # Multiplication of the scene by the coded aperture
    _, M, N, L = y1.shape  # Extract spectral image shape
    # shift and sum
    y2 = prism_operator(y1, shift_sign = 1)
    return y2.sum(dim=1, keepdim=True)


def backward_cassi(y, ca):
    """
    Backward operator of coded aperture snapshot spectral imager (CASSI), more information refer to: Compressive Coded Aperture Spectral Imaging: An Introduction: https://doi.org/10.1109/MSP.2013.2278763
    Args:
        y (torch.Tensor): Measurement with shape (1, M, N + L - 1, 1)
        ca (torch.Tensor): Coded aperture with shape (1, M, N, 1)
    Returns:
        torch.Tensor: Spectral image with shape (1, M, N, L)
    """
    _, _, M, N = y.shape  # Extract spectral image shape
    L = N - M + 1  # Number of shifts
    y = torch.tile(y, [1, L, 1, 1])
    y = prism_operator(y, shift_sign = -1)
    return torch.multiply(y, ca)
