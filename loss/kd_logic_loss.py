# loss/kd_logic_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class KLLogicLoss(nn.Module):
    def __init__(self, temperature=1.0, reduction='batchmean'):
        super(KLLogicLoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, student_logits, teacher_logits):
        T = self.temperature
        return F.kl_div(
            F.log_softmax(student_logits / T, dim=2),
            F.softmax(teacher_logits / T, dim=2),
            reduction=self.reduction
        ) * (T * T)
