import torch

class FilteredSpectralInitialization(torch.nn.Module):
    r"""
    Filtered Spectral Initialization

    This layer implements the Filtered Spectral Initialization method for estimating the optical field from coded measurements of a Coded Phase Imaging System.
    
    Mathematically, this initialization can be described as follows: 

    .... TBD ....

    For more information refer to: Learning spectral initialization for phase retrieval via deep neural networks https://doi.org/10.1364/AO.445085
        
    """
    def __init__(
            self, 
            input_shape:tuple, 
            max_iterations:int , 
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
            max_iterations (int): The maximum number of iterations.
            p (float): The percentage of the top values to keep.
            input_shape (tuple): The shape of the input tensor.

        """
        super(FilteredSpectralInitialization, self).__init__()
        self.iter = max_iterations
        self.p = p 
        self.shape = input_shape 
        self.M = torch.tensor(self.shape[-2] * self.shape[-1], dtype=torch.float32)
        self.R = torch.ceil(self.M / self.p).to(torch.float32)

        if filter is None:
            filter = self.gaussian_kernel(size=k_size, sigma=sigma)
        self.filter = torch.nn.Parameter(filter, requires_grad=train_filter).unsqueeze(0).unsqueeze(0)
        self.filter = self.filter.to(dtype=dtype, device=device)

    def apply_filter(self, data: torch.Tensor, kernel: torch.Tensor)->torch.Tensor:
        return torch.nn.functional.conv2d(data, kernel, padding=kernel.size(-1) // 2)

    def gaussian_kernel(self, size: int, sigma: float)->torch.Tensor:
        x = torch.arange(-size // 2 + 1.0, size // 2 + 1.0)
        x = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = x / x.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        return kernel_2d

    def get_ytr(self, y: torch.Tensor)->torch.Tensor:
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

        
    def forward(self, inputs: torch.Tensor, optical_layer: torch.nn.Module)->torch.Tensor:
        z0 = torch.randn(1, 1, self.shape[-2], self.shape[-1], dtype=torch.float32).to(inputs.device)
        z0 = z0 / torch.norm(z0, p='fro')
        norm_estimation = torch.norm(inputs, p=1, dim=(2,3)) / inputs.numel()**0.5
        Ytr = self.get_ytr(inputs).to(torch.complex64).to(inputs.device)
        div = self.M * self.R
        for _ in range(self.iter):
            y_hat = optical_layer(z0, "forward")
            z0 = torch.div(optical_layer(Ytr * y_hat, "backward"), div)
            Z = self.apply_filter(Z, self.filter)
            z0 = z0 / torch.norm(z0, p='fro', dim=(2, 3))
        z0 = z0 * norm_estimation.to(torch.complex64)
        return z0
    