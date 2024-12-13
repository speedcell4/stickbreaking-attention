import torch
import pytest
import math
from stickbreaking_attention.sb_attn import sb_attn
from transformers import set_seed
from stickbreaking_attention.sb_ref import stickbreaking
from .test_varlen import assert_close


def ref_fwd(q, k, v, length):
    cm = torch.ones(length, length).tril(-1).to(q)
    mask = torch.ones(length, length).triu(0).cuda().bool()
    o, rem = stickbreaking(q, k, v, mask, cm)
    o = o + rem[..., None] * v
    return o

def ref_fwdbwd(do, q, k, v, length):
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    output = ref_fwd(q, k, v, length)
    output.backward(do)
    dq = q.grad
    dk = k.grad
    dv = v.grad
    q.grad = None
    k.grad = None
    v.grad = None
    return output, dq, dk, dv


class TestClass:

    @pytest.mark.parametrize('compile', [False, True])
    @pytest.mark.parametrize('batch_size', [4, 2, 1])
    @pytest.mark.parametrize('num_heads', [24, 8, 4, 2, 1, 7])
    @pytest.mark.parametrize('head_dim', [64, 32, 16, 50])
    @pytest.mark.parametrize('length', [4096, 2048, 1024, 512, 256, 500])
    @pytest.mark.parametrize('dtype', [torch.bfloat16])
    @pytest.mark.parametrize('forward_only', [False])
    def test_varlen(self, batch_size, num_heads, head_dim, length, dtype, forward_only, compile):
        set_seed(1337)
        torch.set_printoptions(linewidth=110, edgeitems=30)
        device = torch.device('cuda:0')
        input_dims = (batch_size, num_heads, length, head_dim)
        v = 0.25 * torch.randn(input_dims, device=device, dtype=torch.float32)
        q = 0.25 * (torch.randn(input_dims, device=device, dtype=torch.float32) + 1) 
        k = 0.25 * (torch.randn(input_dims, device=device, dtype=torch.float32) - 1)
        print(q.max(), k.max(), v.max())
        q = q.to(dtype).requires_grad_()
        k = k.to(dtype).requires_grad_()
        v = v.to(dtype).requires_grad_()
        do = torch.randn(input_dims, device=device, dtype=dtype)
        if compile:
            sb_attn_fun = torch.compile(sb_attn)
        else:
            sb_attn_fun = sb_attn
        with torch.cuda.device(device):
            o, rem= sb_attn_fun(q, k, v, inv_temp=1 / math.sqrt(q.size(-1)))
            o = o + rem[..., None] * v
            ref_out, ref_dq, ref_dk, ref_dv = ref_fwdbwd(do, q, k, v, length)
        eps = 0.05
        torch.cuda.synchronize()
        assert_close("o", ref_out, o, eps)
        if not forward_only:
            dq, dk, dv = torch.autograd.grad(o, inputs=(q, k, v), grad_outputs=do)
            assert_close("dq", ref_dq, dq, eps)
            assert_close("dk", ref_dk, dk, eps)
            assert_close("dv", ref_dv, dv, eps)
