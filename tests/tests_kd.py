import pytest
from .utils import include_colibri

include_colibri()
from torch.utils.data import DataLoader, TensorDataset
from colibri.misc.kd import KD, KD_dec_loss, KD_enc_loss
import torch
from colibri.misc.e2e import E2E
from colibri.train_kd import TrainingKD
from tqdm import tqdm


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


@pytest.fixture
def e2e_teacher_student():
    img_shape = (3, 32, 32)

    from colibri.optics import SD_CASSI

    optical_teacher = SD_CASSI(input_shape=img_shape, trainable=True, binary=False)
    optical_student = SD_CASSI(input_shape=img_shape, trainable=True, binary=True)

    from colibri.models import build_network, Unet_KD

    network_config = dict(
        in_channels=3, out_channels=3, features=[4, 8, 16, 32], last_activation="relu"
    )

    decoder_teacher = build_network(Unet_KD, **network_config)
    decoder_student = build_network(Unet_KD, **network_config)

    teacher = E2E(optical_teacher, decoder_teacher).cuda()
    student = E2E(optical_student, decoder_student).cuda()

    return teacher, student


@pytest.fixture
def kd_instance(e2e_teacher_student):
    teacher, student = e2e_teacher_student

    kd_config = {
        "loss_dec_type": "MSE",
        "loss_enc_type": "GRAMM_SD_CASSI",
        "layer_idxs": [0, 1, 2],
        "dec_weight": 0.5,
        "enc_weight": 0.5,
    }

    return KD(teacher, student, kd_config).cuda()


def test_kd_forward_pass(kd_instance):
    x = torch.randn(4, 3, 32, 32).cuda()
    x_hat_student, loss_dec, loss_enc = kd_instance(x)

    assert x_hat_student.shape == (4, 3, 32, 32), "Student output shape incorrect"
    assert loss_dec.shape == torch.Size([]), "Decoder loss should be a scalar"
    assert loss_enc.shape == torch.Size([]), "Encoder loss should be a scalar"


def test_kd_loss_computation(kd_instance):
    x = torch.randn(4, 3, 32, 32).cuda()
    _, loss_dec, loss_enc = kd_instance(x)

    assert loss_dec.item() >= 0, "Decoder loss should be non-negative"
    assert loss_enc.item() >= 0, "Encoder loss should be non-negative"


def test_kd_invalid_config(e2e_teacher_student):
    teacher, student = e2e_teacher_student

    invalid_kd_config = {
        "loss_dec_type": "INVALID",  # Unsupported loss type
        "loss_enc_type": "GRAMM_SD_CASSI",
        "layer_idxs": [0, 1, 2],
        "dec_weight": 0.5,
        "enc_weight": 0.5,
    }

    with pytest.raises(ValueError):
        KD(teacher, student, invalid_kd_config)


@pytest.fixture
def training_kd_instance(e2e_teacher_student, dummy_dataset, tmp_path):
    teacher, student = e2e_teacher_student
    teacher_path = tmp_path / "teacher.pth"

    torch.save(teacher.state_dict(), teacher_path)  # Save dummy teacher weights

    kd_config = {
        "loss_dec_type": "MSE",
        "loss_enc_type": "GRAMM_SD_CASSI",
        "layer_idxs": [0, 1, 2],
        "dec_weight": 0.5,
        "enc_weight": 0.5,
    }

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

    return TrainingKD(
        student_model=student,
        teacher_model=teacher,
        teacher_path_weights=str(teacher_path),
        train_loader=dummy_dataset,
        optimizer=optimizer,
        kd_config=kd_config,
        device="cuda",
    )


@pytest.fixture
def dummy_dataset():
    """Fixture to create a dummy dataset for training."""
    x = torch.randn(100, 3, 32, 32)  # 100 images (C, H, W)
    dataset = TensorDataset(x)
    return DataLoader(dataset, batch_size=4, shuffle=True)


def test_training_kd_initialization(training_kd_instance):
    assert training_kd_instance.kd_model is not None, "KD model should be initialized"
    assert isinstance(training_kd_instance.kd_model, KD), "kd_model should be an instance of KD"
    assert training_kd_instance.optimizer is not None, "Optimizer should be initialized"


def test_training_kd_missing_teacher_weights(e2e_teacher_student, dummy_dataset):
    teacher, student = e2e_teacher_student

    kd_config = {
        "loss_dec_type": "MSE",
        "loss_enc_type": "GRAMM",
        "layer_idxs": [0, 1, 2],
        "dec_weight": 0.5,
        "enc_weight": 0.5,
    }

    with pytest.raises(ValueError, match="Teacher model weights not found"):
        TrainingKD(
            student_model=student,
            teacher_model=teacher,
            teacher_path_weights=None,  # No weights provided
            train_loader=dummy_dataset,
            kd_config=kd_config,
            device="cuda",
        )
