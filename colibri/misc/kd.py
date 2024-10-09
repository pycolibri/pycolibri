import torch.nn as nn
import torch

class KD(nn.Module):
    def __init__(self, e2e_teacher: nn.Module, e2e_student: nn.Module, loss_type: str):
        r"""
        Knowledge distillation (KD) for computational imaging system design.

        Knowledge distillation module receives a pretrained end-to-end (e2e) teacher model and a e2e student model that is going to be trained.

        """
        super(KD, self).__init__()
        self.e2e_teacher = e2e_teacher
        self.e2e_student = e2e_student
        self.loss_type = loss_type
        self.loss_fn = KD_loss(loss_type)

    def forward(self, x):
        r"""
        Forward pass of the KD model.
        """

        with torch.no_grad():
            x_hat_teacher, feats_teacher = self.e2e_teacher(x)

        x_hat_student, feats_student = self.e2e_student(x)

        loss = self.loss_fn(x_hat_teacher, x_hat_student, feats_teacher, feats_student)

        return loss

        



class KD_loss(nn.Module):
    def __init__(self, loss_type: str):
        r"""
        KD loss function.
        """
        super(KD_loss, self).__init__()
        self.loss_type = loss_type

    def forward(self, x_hat_teacher, x_hat_student, feats_teacher, feats_student):


        if self.loss_type == "L1":
            loss = torch.nn.L1Loss()
            loss_x_hat = loss(x_hat_teacher, x_hat_student)
            return loss_x_hat
        elif self.loss_type == "MSE":
            loss = torch.nn.MSELoss()
            loss_x_hat = loss(x_hat_teacher, x_hat_student)
            return loss_x_hat
        elif self.loss_type == "MSE_Bneck":
            loss = torch.nn.MSELoss()
            loss_bneck = loss(feats_teacher[3], feats_student[3])
        else:
            raise ValueError("Loss type not supported. Please choose between L1 and MSE.")
