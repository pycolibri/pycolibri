import torch.nn as nn
import torch
import torch.nn.functional as F
from colibri.optics.functional import fft


class KD(nn.Module):
    def __init__(
        self,
        e2e_teacher: nn.Module,
        e2e_student: nn.Module,
        kd_config: dict,
    ):
        r"""
        Knowledge Distillation (KD) framework for computational imaging system design.

        This module distills knowledge from a pretrained end-to-end (E2E) teacher model to a
        constrained student model, improving both encoder design and reconstruction performance.
        The teacher system is a less-constrained computational imaging system than the student, achieving high-recovy performance, while the
        student is physically constrained system designed for real-world acquisition.

        The optimization follows:

        .. math::
            \{ \theta_s^*, \learnedOptics_s^* \} = \arg \min_{\learnedOptics_s, \theta_s} \sum_{p=1}^P \lambda_1 \| \mathcal{N}_{\theta_s} (\mathbf{H_{\learnedOptics_s}^\top H_{\learnedOptics_s} \mathbf{x}_p}) - \mathbf{x}_p \|_2^2 + \lambda_2 \mathcal{L}_{\text{DEC}} + \lambda_3 \mathcal{L}_{\text{ENC}}

        where:


        - :math:`\mathcal{L}_{\text{DEC}}` aligns feature representations in the decoder.
        - :math:`\mathcal{L}_{\text{ENC}}` aligns the structure of the student's encoder with the teacher's encder.

        The student is guided by the teacher's encoder :math:`\learnedOptics_t^*` and decoder :math:`\theta_t^*` parameters,
        which are frozen during the training of the student.

        Args:
            e2e_teacher (nn.Module): Pretrained E2E teacher model.
            e2e_student (nn.Module): E2E student model to be trained.
            kd_config (dict): Configuration dictionary containing:
                - "loss_dec_type" (str): Type of decoder loss function.
                - "loss_enc_type" (str): Type of encoder loss function.
                - "layer_idxs" (list): Indices of layers used for decoder loss computation.
                - "att_config" (dict, optional): Additional attention-based configurations.
                - "dec_weight" (float): Weight for decoder loss in KD.
                - "enc_weight" (float): Weight for encoder loss in KD.
        """

        super(KD, self).__init__()
        self.teacher = e2e_teacher
        self.student = e2e_student
        self.kd_config = kd_config
        loss_fb_type = kd_config["loss_dec_type"]
        loss_rb_type = kd_config["loss_enc_type"]
        layer_idxs = kd_config["layer_idxs"]
        self.loss_dec = KD_dec_loss(loss_fb_type, layer_idxs)
        self.loss_enc = KD_enc_loss(loss_rb_type)

    def forward(self, x):
        r"""
        Forward pass of the KD model.
        """

        with torch.no_grad():
            x_hat_teacher, feats_teacher = self.teacher(x)

        x_hat_student, feats_student = self.student(x)

        loss_dec = self.loss_dec(feats_teacher, feats_student) * self.kd_config["dec_weight"]

        loss_enc = (
            self.loss_enc(
                self.teacher.optical_layer.learnable_optics,
                self.student.optical_layer.learnable_optics,
            )
            * self.kd_config["enc_weight"]
        )

        return x_hat_student, loss_dec, loss_enc


class KD_dec_loss(nn.Module):
    def __init__(self, loss_type: str, layer_idxs: list):
        r"""
        KD feature based loss function.
        """
        super(KD_dec_loss, self).__init__()
        self.loss_type = loss_type
        self.layer_idxs = layer_idxs

    def forward(self, feats_teacher, feats_student):

        if self.loss_type == "MSE":
            loss = 0
            for i in self.layer_idxs:
                loss += torch.mean((feats_teacher[i] - feats_student[i]) ** 2)
            return loss / len(self.layer_idxs)

        elif self.loss_type == "L1":
            loss = 0
            for i in self.layer_idxs:
                loss += torch.mean(torch.abs(feats_teacher[i] - feats_student[i]))
            return loss / len(self.layer_idxs)

        elif self.loss_type == "COS":
            loss = 0
            for i in self.layer_idxs:
                loss = loss + 1 - F.cosine_similarity(feats_teacher[i], feats_student[i])
            return loss / len(self.layer_idxs)

        else:
            raise ValueError("Loss type not supported. Please choose between L1, MSE, and COS.")


class KD_enc_loss(nn.Module):
    def __init__(self, loss_type: str):
        r"""
        Knowledge Distillation loss function for aligning the optical encoding parameters
        in the single disperser coded aperture snapshot spectral imager (SD-CASSI) system.

        This loss function ensures that the student’s optical encoding parameters
        :math:`\learnedOptics_s` approximate those of the teacher :math:`\learnedOptics_t^*` by minimizing
        the discrepancy between their Gram matrices.

        The encoder loss function is defined as:

        .. math::
            \mathcal{L}_{\text{enc}} =  \left\| \mathbf{W}_s^\top \mathbf{W}_s - {\learnedOptics_t^*}^\top \learnedOptics_t^* \right\|_F^2

        where:

        - :math:`\mathbf{W}_s` represents the student’s coded aperture parameters (before binarization).
        - :math:`\learnedOptics_t^*` represents the teacher’s optimal coded aperture.


        This loss function encourages structural similarity between the optical encoders of the teacher
        and student by aligning their Gram matrices.

        Args:
            loss_type (str): Type of loss function. Currently supports "GRAMM_SD_CASSI" (Gram matrix-based loss).

        Raises:
            ValueError: If an unsupported loss type is provided.
        """
        super(KD_enc_loss, self).__init__()
        self.loss_type = loss_type

    def forward(self, cas_teacher, cas_student):

        if self.loss_type == "GRAMM_SD_CASSI":

            _, _, M, N = cas_student.shape

            ca_s = cas_student.view(-1, M * N)
            ca_t = cas_teacher.view(-1, M * N)

            gram_s = torch.matmul(ca_s.T, ca_s)
            gram_t = torch.matmul(ca_t.T, ca_t)
            loss = nn.MSELoss()(gram_s, gram_t)
            return loss

        else:
            raise ValueError("Loss type not supported. Please choose between GRAMM_SD_CASSI.")
