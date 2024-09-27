import torch.nn as nn


class KD(nn.Module):
    def __init__(self, e2e_teacher: nn.Module, e2e_student: nn.Module):
        r"""
        Knowledge distillation (KD) for computational imaging system design.
        """
        super(KD, self).__init__()
        self.e2e_teacher = e2e_teacher
        self.e2e_student = e2e_student

    def forward(self, x):
        r"""
        Forward pass of the KD model.

        """
        x_hat_teacher, feats_teacher = self.e2e_teacher(x)
        x_hat_student, feats_student = self.e2e_student(x)
        return [x_hat_teacher, x_hat_student], [feats_teacher, feats_student]
