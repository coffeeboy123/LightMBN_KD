# loss/kd_logic_loss_qaware.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class KLLogicLossQAware(nn.Module):
    """
    QAT-aware KL: 양자화 오차(전/후 로짓 차이)로 KL을 가중.
    - s_post: FakeQuant 통과 후 로짓 (B,H,C)
    - t_logits: Teacher 로짓 (B,H,C)
    - s_pre: FakeQuant 통과 전 로짓 (B,H,C)
    """
    def __init__(self, temperature: float = 2.0,
                 alpha: float = 1.0,         # 오차를 얼마나 증폭할지
                 eps: float = 1e-6,
                 w_min: float = 1.0,
                 w_max: float = 5.0,
                 norm: str = "l2",           # "l1" or "l2"
                 per_sample: bool = False,   # True면 샘플별-헤드별 가중, False면 헤드별 가중(배치 평균)
                 detach_error: bool = True):
        super().__init__()
        self.T = float(temperature)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.norm = norm
        self.per_sample = per_sample
        self.detach_error = detach_error

    def _compute_error_weight(self, s_post, s_pre):
        # s_post, s_pre: (B,H,C)
        if self.norm == "l1":
            err = (s_post - s_pre).abs().sum(dim=-1)    # (B,H)
        else:  # l2
            err = torch.norm(s_post - s_pre, p=2, dim=-1)  # (B,H)

        if self.detach_error:
            err = err.detach()

        # 크기 정규화(안정화): 배치/헤드 단위 평균으로 스케일 다운
        denom = err.mean().clamp_min(self.eps)  # 스칼라
        norm_err = err / denom                  # (B,H)

        w = 1.0 + self.alpha * norm_err         # (B,H)

        # per-sample이 아니면 헤드별 평균 스칼라로 축소
        if not self.per_sample:
            w = w.mean(dim=0, keepdim=True)     # (1,H)

        # 클램핑
        w = w.clamp_(self.w_min, self.w_max)    # (B,H) or (1,H)
        return w

    def forward(self, s_post: torch.Tensor, t_logits: torch.Tensor, s_pre: torch.Tensor):
        """
        s_post, t_logits, s_pre: (B,C) or (B,H,C)
        """
        T = self.T

        # (B,C) -> (B,1,C)
        def _u3(x):
            return x.unsqueeze(1) if x.dim() == 2 else x

        s_post = _u3(s_post)
        t_logits = _u3(t_logits)
        s_pre = _u3(s_pre)

        B, H, C = s_post.shape
        assert t_logits.shape == (B, H, C) and s_pre.shape == (B, H, C), "Shape mismatch"

        # 헤드별 가중치 계산
        w = self._compute_error_weight(s_post, s_pre)  # (B,H) or (1,H)

        with torch.no_grad():
            pt = torch.softmax(t_logits / T, dim=-1)   # (B,H,C)

        log_ps = torch.log_softmax(s_post / T, dim=-1) # (B,H,C)

        kl_all = F.kl_div(log_ps, pt, reduction="none")  # (B,H,C)
        kl_all = kl_all.sum(dim=-1)                      # (B,H) 클래스 합

        # 가중 적용
        if w.shape[0] == 1:  # 헤드별 스칼라
            kl_all = kl_all * w          # (B,H)
        else:                # per-sample
            kl_all = kl_all * w

        # 배치 평균 후 헤드 합산
        kl_per_head = kl_all.mean(dim=0)    # (H,)
        loss = kl_per_head.sum() * (T * T)  # T^2 보정은 유지
        return loss
