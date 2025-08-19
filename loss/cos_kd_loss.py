# loss/cosine_kd_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineKDLoss(nn.Module):
    """
    inputs:
      student: (B, Cs) or (B, Cs, T)
      teacher: (B, Ct) or (B, Ct, T)
    전제: Engine에서 student를 proj로 teacher 차원(Ct)으로 맞춰 넘겨줌.
    계산: 채널(C) 축 정규화 후 같은 위치에서 cos 유사도 → (1 - cos) 평균
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        if student.dim() == 2:
            # (B,C) -> (B,C,1)
            student = student.unsqueeze(-1)
            teacher = teacher.unsqueeze(-1)

        B, C, T = student.shape
        assert teacher.shape == (B, C, T), f"shape mismatch: {student.shape} vs {teacher.shape}"

        s = F.normalize(student, p=2, dim=1)  # (B,C,T)
        t = F.normalize(teacher, p=2, dim=1)  # (B,C,T)

        cos = (s * t).sum(dim=1)             # (B,T)
        loss = 1.0 - cos                     # (B,T)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
