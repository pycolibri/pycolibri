import pytest
from .utils import include_colibri

include_colibri()

from colibri.misc.kd import KD, KD_dec_loss, KD_enc_loss
import torch


@pytest.fixture
def feature_maps():
    batch_size, num_layers, channels, height, width = 2, 3, 16, 32, 32
    feats_teacher = [torch.randn(batch_size, channels, height, width) for _ in range(num_layers)]
    feats_student = [torch.randn(batch_size, channels, height, width) for _ in range(num_layers)]
    return feats_teacher, feats_student


@pytest.fixture
def coded_apertures():
    batch_size, channels, height, width = 1, 4, 32, 32
    cas_teacher = torch.randn(batch_size, channels, height, width)
    cas_student = torch.randn(batch_size, channels, height, width)
    return cas_teacher, cas_student


def test_kd_dec_loss(feature_maps):
    feats_teacher, feats_student = feature_maps
    kd_loss = KD_dec_loss(loss_type="MSE", layer_idxs=[0, 1, 2])
    loss = kd_loss(feats_teacher, feats_student)
    assert loss.shape == torch.Size([]), "Loss should be a scalar"


def test_kd_enc_loss(coded_apertures):
    cas_teacher, cas_student = coded_apertures
    kd_loss = KD_enc_loss(loss_type="GRAMM_SD_CASSI")
    loss = kd_loss(cas_teacher, cas_student)
    assert loss.shape == torch.Size([]), "Loss should be a scalar"
