import torch_dct as dct


class DCT2D:
    def __init__(self, norm='ortho'):
        self.norm = norm

    def forward(self, x):
        return dct.dct_2d(x, norm=self.norm)

    def inverse(self, x):
        return dct.idct_2d(x, norm=self.norm)