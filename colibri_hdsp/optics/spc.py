import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class SPC(nn.Module):

    def __init__(self, img_size=32, m=256):
        """
        Initializes the Single Pixel Camera (SPC) model.

        Args:
            img_size (int): Size of the image. Default is 32.
            m (int): Number of measurements. Default is 256.
        """
        super(SPC, self).__init__()

        self.H = nn.Parameter(torch.randn(m, img_size**2))
        self.image_size = img_size

    def forward(self, x):
        """
        Forward propagation through the SPC model.

        Args:
            x (torch.Tensor): Input image tensor of size (b, c, h, w).

        Returns:
            torch.Tensor: Output tensor after measurement.
        """

        # spatial vectorization
        b, c, h, w = x.size()
        x = x.view(b, c, h*w)
        x = x.permute(0, 2, 1)

        # measurement
        H = self.H.unsqueeze(0).repeat(b, 1, 1)
        y = torch.bmm(H, x)
        return y


    def inverse(self, y):
        """
        Inverse operation to reconstruct the image from measurements.

        Args:
            y (torch.Tensor): Measurement tensor of size (b, m, c).

        Returns:
            torch.Tensor: Reconstructed image tensor.
        """

        Hinv = torch.pinverse(self.H)
        Hinv = Hinv.unsqueeze(0).repeat(y.shape[0], 1, 1)

        x = torch.bmm(Hinv, y)
        x = x.permute(0, 2, 1)
        b, c, hw = x.size()
        h = int(np.sqrt(hw))
        x = x.view(b, c, h, h)
        return x

    def get_weights(self):
        """
        Gets the measurement matrix.

        Returns:
            torch.Tensor: Flattened measurement matrix.
        """
        return torch.flatten(self.H)
    

if __name__ == "__main__":
    import torchvision
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt


    train_set = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    image, label = train_set[0]
    image = image.unsqueeze(0)

    model = SPC()
    y = model(image)
    x_hat = model.inverse(y)

    plt.subplot(1, 2, 1)
    plt.imshow(image.squeeze().permute(1, 2, 0))
    plt.title('original')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.imshow(x_hat.squeeze().permute(1, 2, 0).detach())
    plt.title('reconstruction')
    plt.xticks([])
    plt.yticks([])

    plt.show()
