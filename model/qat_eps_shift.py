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

    def _pre_hook(mod: FakeQuantize, inputs):
        x = inputs[0]
        # eps를 (-0.5, 0.5) 안에 유지 (안정)
        e = torch.tanh(mod.eps) * 0.5

        # scale은 per-tensor(스칼라)거나 per-channel(텐서)일 수 있음
        s = mod.scale
        if s is None:
            return inputs  # 아직 observer가 scale 못 만들었으면 패스

        # 브로드캐스트 처리(대부분 activation은 per-tensor라 그냥 곱해짐)
        try:
            shift = e * s
            return (x + shift, )
        except RuntimeError:
            # per-channel weight 같은 특수 케이스 대비(필요시 확장)
            return (x + e * s.view(1, -1, 1, 1), )

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
