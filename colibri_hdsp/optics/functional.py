import torch
import numpy as np

def prism_operator(x, shift_sign = 1):
    """
    Prism operator, it shifts the input tensor x according to spectral shift made by a prism
    Args:
        x (torch.Tensor): Input tensor with shape (1, L, M, N)
        shift_sign (int): Integer, it can be 1 or -1, it indicates the direction of the shift
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
    """
    Forward operator of color coded aperture snapshot spectral imager (Color-CASSI), more information refer to: Computational snapshot multispectral cameras: Toward dynamic capture of the spectral world https://doi.org/10.1109/MSP.2016.2582378
    Args:
        x (torch.Tensor): Spectral image with shape (1, L, M, N)
        ca (torch.Tensor): Coded aperture with shape (1, L, M, N)
    
    Returns: 
        torch.Tensor: Measurement with shape (1, 1, M, N)
    """
    y = torch.multiply(x, ca)
    return y.sum(dim=1, keepdim=True)

def backward_color_cassi(y, ca):
    """
    Backward operator of color coded aperture snapshot spectral imager (Color-CASSI), more information refer to: Computational snapshot multispectral cameras: Toward dynamic capture of the spectral world https://doi.org/10.1109/MSP.2016.2582378
    Args:
        y (torch.Tensor): Measurement with shape (1, 1, M, N)
        ca (torch.Tensor): Coded aperture with shape (1, L, M, N)
    Returns:
        torch.Tensor: Spectral image with shape (1, L, M, N)
    """
    x = torch.multiply(y, ca)
    return x


def forward_dd_cassi(x, ca):
    """
    Forward operator of dual disperser coded aperture snapshot spectral imager (DD-CASSI), more information refer to: Computational snapshot multispectral cameras: Toward dynamic capture of the spectral world https://doi.org/10.1109/MSP.2016.2582378
    Args:
        x (torch.Tensor): Spectral image with shape (1, L, M, N)
        ca (torch.Tensor): Coded aperture with shape (1, 1, M, N + L - 1)
    Returns:
        torch.Tensor: Measurement with shape (1, 1, M, N + L - 1)
    """
    _, L, M, N = x.shape  # Extract spectral image shape
    assert ca.shape[-1] == N + L - 1, "The coded aperture must have the same size as a dispersed scene"
    ca = torch.tile(ca, [1, L, 1, 1])
    ca = prism_operator(ca, shift_sign = -1)
    y = forward_color_cassi(x, ca)
    return y


def backward_dd_cassi(y, ca):
    """
    Backward operator of dual disperser coded aperture snapshot spectral imager (DD-CASSI), more information refer to: Computational snapshot multispectral cameras: Toward dynamic capture of the spectral world https://doi.org/10.1109/MSP.2016.2582378
    Args:
        y (torch.Tensor): Measurement with shape (1, 1, M, N + L - 1)
        ca (torch.Tensor): Coded aperture with shape (1, 1, M, N + L - 1)
    Returns:
        torch.Tensor: Spectral image with shape (1, L, M, N)
    """
    _, M, N, _ = ca.shape  # Extract spectral image shape
    L = N - M + 1  # Number of shifts
    y = torch.tile(y, [1, L, 1, 1])
    ca = torch.tile(ca, [1, L, 1, 1])
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
        y (torch.Tensor): Measurement with shape (1, 1, M, N + L - 1)
        ca (torch.Tensor): Coded aperture with shape (1, 1, M, N)
    Returns:
        torch.Tensor: Spectral image with shape (1, L, M, N)
    """
    _, _, M, N = y.shape  # Extract spectral image shape
    L = N - M + 1  # Number of shifts
    y = torch.tile(y, [1, L, 1, 1])
    y = prism_operator(y, shift_sign = -1)
    return torch.multiply(y, ca)


def forward_spc(x, H):
    """
    Forward propagation through the SPC model.

    Args:
        x (torch.Tensor): Input image tensor of size (b, c, h, w).
        H (torch.Tensor): Measurement matrix of size (m, h*w).

    Returns:
        torch.Tensor: Output tensor after measurement.
    """

    b, c, h, w = x.size()
    x = x.reshape(b, c, h*w)
    x = x.permute(0, 2, 1)

    # measurement
    H = H.unsqueeze(0).repeat(b, 1, 1)
    y = torch.bmm(H, x)
    return y

def backward_spc(y, H):
    """
    Inverse operation to reconsstruct the image from measurements.

    Args:
        y (torch.Tensor): Measurement tensor of size (b, m, c).

    Returns:
        torch.Tensor: Reconstructed image tensor.
    """

    Hinv = torch.pinverse(H)
    Hinv = Hinv.unsqueeze(0).repeat(y.shape[0], 1, 1)

    x = torch.bmm(Hinv, y)
    x = x.permute(0, 2, 1)
    b, c, hw = x.size()
    h = int(np.sqrt(hw))
    x = x.reshape(b, c, h, h)
    return x