# qat_eps_shift.py
import torch
import torch.nn as nn
from torch.quantization.fake_quantize import FakeQuantize

def attach_rounding_eps_shift(model: nn.Module, activation_only: bool = True):
    handles, num = [], 0

    def _pre_hook(mod, args):
        (x,) = args

        # 1) scale (버퍼) → detach + device + (필요시) clone된 broadcast 텐서
        s = getattr(mod, "scale", None)
        if s is None:
            return (x,)
        s_safe = s.detach().to(x.device)           # 그래프 끊기
        if s_safe.dim() == 0:
            s_expand = s_safe
        else:
            view_shape = [1] * x.dim()
            ch_axis = getattr(mod, "ch_axis", 1 if x.dim() >= 2 else 0)
            ch_axis = (ch_axis + x.dim()) % x.dim()
            view_shape[ch_axis] = -1
            # ★ 원본 버퍼 공유 금지: reshape 후 clone으로 독립 버퍼
            s_expand = s_safe.reshape(*view_shape).clone()

        # 2) 학습 파라미터 eps
        e = getattr(mod, "eps", None)
        if e is None:
            return (x,)
        e_eff = 0.49 * torch.tanh(e)               # in-place 없음

        # 3) shift 만들고 dtype/device 맞추기
        shift = (e_eff * s_expand).to(dtype=x.dtype, device=x.device)

        # 4) 입력은 반드시 새 텐서로 반환 (in-place 금지)
        x_mod = (x + shift).contiguous()           # clone() 대신 contiguous()도 OK
        return (x_mod,)

    for name, m in model.named_modules():
        if isinstance(m, FakeQuantize):
            if activation_only and ("activation_post_process" not in name):
                continue
            if not hasattr(m, "eps"):
                m.register_parameter("eps", nn.Parameter(torch.zeros(1)))
            if not hasattr(m, "_eps_shift_hook"):
                h = m.register_forward_pre_hook(_pre_hook)
                m._eps_shift_hook = h
                handles.append(h)
                num += 1

    return handles, num


def freeze_eps(model: nn.Module, requires_grad: bool = False):
    for m in model.modules():
        if isinstance(m, FakeQuantize) and hasattr(m, "eps"):
            m.eps.requires_grad_(requires_grad)
