import math
import torch
import triton.language as tl
from torch.nn import functional as F

log2 = math.log(2)
inv_log2 = 1 / log2
ALLOW_TF32 = True

from .sb_varlen_fwd import sb_fwd
from .sb_varlen_bwd import sb_bwd

class StickBreakingAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens, inv_temp):
        no_grad = not ctx.needs_input_grad[0]
        logit_scale = inv_temp

        o, rem, neg_log_acc, seq_program_offsets = sb_fwd(
            q, k, v, cu_seqlens,
            logit_scale=inv_temp,
            no_grad=no_grad
        )
        ctx.save_for_backward(q, k, v, neg_log_acc, cu_seqlens, seq_program_offsets)
        ctx.logit_scale = logit_scale
        return o, rem

    @staticmethod
    def backward(ctx, do, drem):
        logit_scale = ctx.logit_scale
        q, k, v, neg_log_acc, cu_seqlens, seq_program_offsets = ctx.saved_tensors
        dq, dk, dv = sb_bwd(
            do, drem, q, k, v, cu_seqlens, seq_program_offsets, neg_log_acc, logit_scale
        )
        return dq, dk, dv, None, None


def sb_attn_varlen(q, k, v, cu_seqlens, inv_temp=None, zero_start=True):
    if zero_start:
        assert cu_seqlens[0] == 0
        cu_seqlens = cu_seqlens[1:]
    if inv_temp is None:
        inv_temp = 1 / math.sqrt(q.size(-1))

    return sb_attn_varlen_(q, k, v, inv_temp, cu_seqlens)


def sb_attn_varlen_(q, k, v, inv_temp, cu_seqlens):
    return StickBreakingAttention.apply(q, k, v, cu_seqlens, inv_temp)
