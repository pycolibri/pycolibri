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
        Knowledge distillation (KD) for computational imaging system design.

        Knowledge distillation module receives a pretrained end-to-end (e2e) teacher model and a e2e student model that is going to be trained.

        """
        super(KD, self).__init__()
        self.teacher = e2e_teacher
        self.student = e2e_student
        self.kd_config = kd_config
        loss_fb_type = kd_config["loss_fb_type"]
        loss_rb_type = kd_config["loss_rb_type"]
        layer_idxs = kd_config["layer_idxs"]
        att_config = kd_config["att_config"]
        self.loss_fb = KD_fb_loss(loss_fb_type, layer_idxs, att_config)
        self.loss_rb = KD_rb_loss(loss_rb_type)

    def forward(self, x):
        r"""
        Forward pass of the KD model.
        """

        with torch.no_grad():
            x_hat_teacher, feats_teacher = self.teacher(x)

        x_hat_student, feats_student = self.student(x)

        loss_fb = self.loss_fb(feats_teacher, feats_student) * self.kd_config["fb_weight"]

        loss_rb = self.loss_rb(x_hat_teacher, x_hat_student) * self.kd_config["rb_weight"]

        return x_hat_student, loss_fb, loss_rb


class KD_fb_loss(nn.Module):
    def __init__(self, loss_type: str, layer_idxs: list, att_config: dict = None):
        r"""
        KD feature based loss function.
        """
        super(KD_fb_loss, self).__init__()
        self.loss_type = loss_type
        self.layer_idxs = layer_idxs
        self.att_config = att_config

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

        elif self.loss_type == "ATT" and self.att_config:
            param = self.att_config["param"]
            exp = self.att_config["exp"]
            norm = self.att_config["norm"]

            loss = 0
            for i in self.layer_idxs:
                attention_map_teacher = self.get_attention(feats_teacher[i], param, exp, norm)
                attention_map_student = self.get_attention(feats_student[i], param, exp, norm)

                loss += torch.mean((attention_map_teacher - attention_map_student) ** 2)

            return loss / len(self.layer_idxs)

        else:
            raise ValueError("Loss type not supported. Please choose between L1, MSE, COS and ATT.")

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
            real_part = torch.mean((fft(x_hat_teacher).real - fft(x_hat_student).real) ** 2)
            imag_part = torch.mean((fft(x_hat_teacher).imag - fft(x_hat_student).imag) ** 2)
            return (real_part + imag_part) / 2
        
        else:
            raise ValueError("Loss type not supported. Please choose between L1 and MSE.")
