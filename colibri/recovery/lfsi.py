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
        data_real, data_imag = data.real, data.imag
        data_real = torch.nn.functional.conv2d(data_real, kernel, padding=kernel.size(-1) // 2)
        data_imag = torch.nn.functional.conv2d(data_imag, kernel, padding=kernel.size(-1) // 2)
        return data_real + 1j * data_imag
        return data_imag#

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
        z0 = torch.randn_like(inputs)
        z0 = z0 / torch.norm(z0, p='fro')
        norm_estimation = torch.norm(inputs, p=1, dim=(-2,-1)) / inputs.numel()**0.5
        Ytr = self.get_ytr(inputs).to(torch.complex64).to(inputs.device)
        div = self.M * self.R
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(torch.abs(inputs[0]).cpu().detach().numpy())
        ax[0].set_title("Input")
        ax[1].imshow(torch.abs(Ytr[0]).cpu().detach().numpy())
        ax[1].set_title("Ytr")
        plt.show()

        
        for _ in range(self.iter):
            y_hat = optical_layer(z0, "forward", intensity=False)
            z0 = torch.div(optical_layer(Ytr * y_hat, "backward", intensity=False), div)
            z0 = self.apply_filter(z0, self.filter)
            z0 = z0 / torch.norm(z0, p='fro', dim=(-2, -1))
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(torch.abs(z0[0]).cpu().detach().numpy())
            ax[0].set_title("Reconstruction")
            ax[1].imshow(torch.angle(z0[0]).cpu().detach().numpy())
            ax[1].set_title("Phase")
            plt.show()

        z0 = z0 * norm_estimation.to(torch.complex64)
        return z0
    