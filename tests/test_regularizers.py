import pytest
from .utils import include_colibri
include_colibri()

from colibri_hdsp import regularizers
import torch

reg_list = ["binary", "transmittance", "correlation", "kl_gaussian", "min_variance"]  

@pytest.fixture
def imsize():
    b = 16
    h = 128
    w = 128
    c = 3
    return b, c, h, w

def choose_regularizer(name, imsize):

    if name == "binary":
        reg = regularizers.Reg_Binary()
    elif name == "transmittance":
        reg = regularizers.Reg_Transmittance(t=0.1)
    elif name == "correlation":
        reg = regularizers.Correlation()
    elif name == "kl_gaussian":
        reg = regularizers.KLGaussian(mean=torch.Tensor([0.0]),stddev=torch.Tensor([1.0]))
    elif name == "min_variance":
        reg = regularizers.MinVariance()
    return reg

@pytest.mark.parametrize("reg_name", reg_list)
def test_model(reg_name, imsize):
    
    reg = choose_regularizer(reg_name, imsize)
    if reg_name=="binary":
        x1 = torch.randint(0, 2, imsize)
        x2 = torch.rand(imsize)
        cond = reg(x1)<reg(x2)
    elif reg_name=="transmittance":
        x1 = torch.round(torch.rand(imsize)+0.1)
        x2 = torch.round(torch.rand(imsize)+0.8)
        cond = reg(x1)<reg(x2)
    elif reg_name=="correlation":
        x1 = torch.rand(128, 3, 256, 256)
        x2 = torch.rand(128, 3, 256, 256)
        cond = reg((x1,x2))>reg((x2,x2))
    elif reg_name=="kl_gaussian":
        x1 = torch.normal(0,1,size=imsize)
        x2 = torch.normal(0,5,size=imsize)
        cond = reg(x1)<reg(x2)
    elif reg_name=="min_variance":
        x1 = torch.normal(0,1,size=imsize)
        x2 = torch.normal(0,5,size=imsize)
        cond = reg(x1)<reg(x2)

    assert  cond