import torch.nn as nn
import torch
import torch.nn.functional as F
from colibri.optics.functional import fft


class KD(nn.Module):
    def __init__(
        self,
        e2e_teacher: nn.Module,
        e2e_student: nn.Module,
        loss_fb_type: str,
        ft_idx: int,
        loss_rb_type: str,
        att_config: dict = None,
    ):
        r"""
        Knowledge distillation (KD) for computational imaging system design.

        Knowledge distillation module receives a pretrained end-to-end (e2e) teacher model and a e2e student model that is going to be trained.

        """
        super(KD, self).__init__()
        self.e2e_teacher = e2e_teacher
        self.e2e_student = e2e_student
        self.loss_fb = KD_fb_loss(loss_fb_type, ft_idx, att_config)
        self.loss_rb = KD_rb_loss(loss_rb_type)

    def forward(self, x):
        r"""
        Forward pass of the KD model.
        """

        with torch.no_grad():
            x_hat_teacher, feats_teacher = self.e2e_teacher(x)

        x_hat_student, feats_student = self.e2e_student(x)

        loss_fb = self.loss_fb(x_hat_teacher, x_hat_student, feats_teacher, feats_student)

        loss_rb = self.loss_rb(x_hat_teacher, x_hat_student)

        return x_hat_student, loss_fb, loss_rb


class KD_fb_loss(nn.Module):
    def __init__(self, loss_type: str, ft_idx: int, att_config: dict = None):
        r"""
        KD feature based loss function.
        """
        super(KD_fb_loss, self).__init__()
        self.loss_type = loss_type
        self.ft_idx = ft_idx
        self.att_config = att_config

    def forward(self, feats_teacher, feats_student):

        if self.loss_type == "MSE":
            return torch.mean((feats_teacher[self.ft_idx] - feats_student[self.ft_idx]) ** 2)

        elif self.loss_type == "L1":
            return torch.mean(torch.abs(feats_teacher[self.ft_idx] - feats_student[self.ft_idx]))

        elif self.loss_type == "ATT" and self.att_config:
            param = self.att_config["param"]
            exp = self.att_config["exp"]
            norm = self.att_config["norm"]

            attention_map_teacher = self.get_attention(feats_teacher[self.ft_idx], param, exp, norm)
            attention_map_student = self.get_attention(feats_student[self.ft_idx], param, exp, norm)

            return torch.mean((attention_map_teacher - attention_map_student) ** 2)

        else:
            raise ValueError("Loss type not supported. Please choose between L1, MSE and ATT.")

    def get_attention(feature_set, param=0, exp=4, norm="l2"):
        # Adapted from:
        # Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer https://arxiv.org/abs/1612.03928
        # https://github.com/DataDistillation/DataDAM/blob/main/utils.py
        if param == 0:
            attention_map = torch.sum(torch.abs(feature_set), dim=1)

        elif param == 1:
            attention_map = torch.sum(torch.abs(feature_set) ** exp, dim=1)

        elif param == 2:
            attention_map = torch.max(torch.abs(feature_set) ** exp, dim=1)

        if norm == "l2":

            vectorized_attention_map = attention_map.view(feature_set.size(0), -1)
            normalized_attention_maps = F.normalize(vectorized_attention_map, p=2.0)

        elif norm == "fro":

            un_vectorized_attention_map = attention_map

            fro_norm = torch.sum(torch.sum(torch.abs(attention_map) ** 2, dim=1), dim=1)

            normalized_attention_maps = un_vectorized_attention_map / fro_norm.unsqueeze(
                dim=-1
            ).unsqueeze(dim=-1)
        elif norm == "l1":

            vectorized_attention_map = attention_map.view(feature_set.size(0), -1)
            normalized_attention_maps = F.normalize(vectorized_attention_map, p=1.0)

        elif norm == "none":
            normalized_attention_maps = attention_map

        elif norm == "none-vectorized":
            normalized_attention_maps = attention_map.view(feature_set.size(0), -1)

        return normalized_attention_maps


class KD_rb_loss(nn.Module):
    def __init__(self, loss_type: str):
        r"""
        KD response based loss function.
        """
        super(KD_rb_loss, self).__init__()
        self.loss_type = loss_type

    def forward(self, x_hat_teacher, x_hat_student):

        if self.loss_type == "MSE":
            return torch.mean((x_hat_teacher - x_hat_student) ** 2)
        elif self.loss_type == "L1":
            return torch.mean(torch.abs(x_hat_teacher - x_hat_student))
        elif self.loss_type == "FFT":
            return torch.mean(torch.abs(fft(x_hat_teacher) - fft(x_hat_student)) ** 2)
        else:
            raise ValueError("Loss type not supported. Please choose between L1 and MSE.")
