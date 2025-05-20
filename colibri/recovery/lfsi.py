import torch

class LFSI(torch.nn.Module):
    r"""
    Learned Filtered Spectral Initialization (LFSI).

    This layer implements the Learned Filtered Spectral Initialization method for estimating the optical field from coded measurements of a Coded Phase Imaging System.
    
    This algorithm approximates an initial guess of the optical field :math:`\mathbf{x}^{(0)}` by computing a filtered version of the leading eigenvector of the sensing matrix taking into account the measurements. For more information refer to: Learning spectral initialization for phase retrieval via deep neural networks https://doi.org/10.1364/AO.445085
        
    """
    def __init__(
            self, 
            max_iters:int , 
            filter: torch.Tensor = None,
            p: float = 0.6,
            k_size: int = 5, 
            sigma: float = 1.0, 
            train_filter: bool = False,
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device("cpu")
        )->None:

        r"""
        Initializes the Filtered Spectral Initialization layer.

        Args:
            max_iters (int): The maximum number of iterations.
            filter (torch.Tensor, optional): The filter tensor. If None, a Gaussian kernel will be used. Default is None.
            p (float, optional): The percentage of the top values to keep. Default is 0.6.
            k_size (int, optional): The size of the Gaussian kernel. Default is 5.
            sigma (float, optional): The standard deviation of the Gaussian kernel. Default is 1.0.
            train_filter (bool, optional): If True, the filter will be trainable. Default is False.
            dtype (torch.dtype, optional): The data type of the filter tensor. Default is torch.float32.
            device (torch.device, optional): The device on which the filter tensor will be allocated. Default is CPU.

            Returns:
            None
        """
        super(LFSI, self).__init__()
        self.iter = max_iters
        self.p = p 


        if filter is None:
            filter = self.gaussian_kernel(size=k_size, sigma=sigma)
        self.filter = torch.nn.Parameter(filter, requires_grad=train_filter).unsqueeze(0).unsqueeze(0)
        self.filter = self.filter.to(dtype=dtype, device=device)

    def apply_filter(self, data: torch.Tensor, kernel: torch.Tensor)->torch.Tensor:
        data_real, data_imag = data.real, data.imag
        data_real = torch.nn.functional.conv2d(data_real, kernel, padding=kernel.size(-1) // 2)
        data_imag = torch.nn.functional.conv2d(data_imag, kernel, padding=kernel.size(-1) // 2)
        return data_real + 1j * data_imag

    def gaussian_kernel(self, size: int, sigma: float)->torch.Tensor:
        x = torch.arange(-size // 2 + 1.0, size // 2 + 1.0)
        x = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = x / x.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        return kernel_2d

    def get_ytr(self, y: torch.Tensor)->torch.Tensor:
        r"""

        This function generates a mask that keeps the top p% of the values of the input tensor.

        Args:
            y (torch.Tensor): The input tensor.
        
        Returns:
            torch.Tensor: The mask tensor
        """    
        y = (y-y.min())/(y.max()-y.min())
        ytr_f = torch.where(y.abs() >= (1-self.p)*y.max(), torch.ones_like(y), torch.zeros_like(y))
        return ytr_f

        
    def forward(self, inputs: torch.Tensor, optical_layer: torch.nn.Module)->torch.Tensor:
        r"""
        
        This function estimates the optical field from coded measurements of a Coded Phase Imaging System using the Filtered Spectral Initialization method.

        Args:
            inputs (torch.Tensor): The input tensor.
            optical_layer (torch.nn.Module): The optical layer that represents the Coded Phase Imaging System.
        
        Returns:
            torch.Tensor: The estimated optical field.
        """

        z0 = torch.randn_like(inputs)

        z0 = z0 / torch.norm(z0, p='fro', dim=(-2, -1), keepdim=True)
        norm_estimation = torch.norm(inputs, p=1, dim=(-2,-1)) / inputs.numel()**0.5
        Ytr = self.get_ytr(inputs).to(torch.complex64).to(inputs.device)

        for _ in range(self.iter):
            y_hat = optical_layer(z0, "forward", intensity=False)
            z0 = optical_layer(Ytr * y_hat, "backward", intensity=False)
            z0 = self.apply_filter(z0, self.filter)
            z0 = z0 / torch.norm(z0, p='fro', dim=(-2, -1))

        z0 = z0 * norm_estimation.to(torch.complex64)
        return z0
    