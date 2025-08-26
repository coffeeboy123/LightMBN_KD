# qat_eps_shift.py
import torch
import torch.nn as nn
from torch.quantization.fake_quantize import FakeQuantize

def attach_rounding_eps_shift(model: nn.Module, activation_only: bool = True):
    """
    모든(또는 activation 전용) FakeQuantize 모듈에
    y = FakeQuant( x + eps * scale ) 를 주입.
    - eps: 모듈별 학습 파라미터 (초기 0)
    - scale: 해당 FakeQuant 모듈이 관측기로 추정한 scale 버퍼
    반환: (hook_handles, num_hooked)
    """
    handles = []
    num = 0

    def _pre_hook(mod, args):
        (x,) = args

        # 1) 안전한 scale: detach + device 맞추기
        ap = getattr(mod, "activation_post_process", None)
        if ap is None or not hasattr(ap, "scale"):
            return (x, )

        s = ap.scale
        # s: per-tensor([]) 혹은 per-channel([C]) 텐서
        s_safe = s.detach().to(x.device)  # ← 버전 고정본

        # 2) epsilon도 in-place 금지: 파생 텐서로 사용
        e = getattr(mod, "rounding_eps", None)
        if e is None:
            return (x, )

        # 부드럽고 유한한 경계: tanh 로 -0.49~0.49 범위
        e_eff = 0.49 * torch.tanh(e)      # ← 새 텐서, in-place 없음

        # 3) shape broadcast (per-tensor vs per-channel)
        if s_safe.dim() == 0:
            shift = e_eff * s_safe        # scalar
        else:
            # per-channel fake-quant (C,) → (1,C,1,1)로 맞춤
            shift = e_eff * s_safe.view(1, -1, 1, 1)

    # 4) x는 절대 in-place 금지 (x += ... X)
        return (x + shift, )

    for name, m in model.named_modules():
        if isinstance(m, FakeQuantize):
            if activation_only and ("activation_post_process" not in name):
                continue
            if not hasattr(m, "eps"):
                # FakeQuant 모듈 안에 학습 파라미터 주입
                m.register_parameter("eps", nn.Parameter(torch.zeros(1)))
            # 이미 훅 달려있으면 중복 방지
            if not hasattr(m, "_eps_shift_hook"):
                h = m.register_forward_pre_hook(_pre_hook)
                m._eps_shift_hook = h
                handles.append(h)
                num += 1

    return handles, num


def freeze_eps(model: nn.Module, requires_grad: bool = False):
    """학습 중/후 eps 파라미터를 켜고/끄는 유틸"""
    for m in model.modules():
        if isinstance(m, FakeQuantize) and hasattr(m, "eps"):
            m.eps.requires_grad_(requires_grad)
