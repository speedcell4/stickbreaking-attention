import torch
import pytest
import math
from torch.nn import functional as F
from stickbreaking_attention.sb_attn import sb_attn
import triton
from flash_attn import flash_attn_func
from flash_attn.flash_attn_triton import flash_attn_func as triton_flash_attn_func
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb, rotate_half
from transformers import set_seed


def tri_fwdbwd(do, q, k, v):
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    o, rem = sb_attn(q, k, v, inv_temp=1 / math.sqrt(q.size(-1)))
    o = o.permute(0, 2, 1, 3)
    # o = o + rem[..., None] * v
    return o

def flash_fwdbwd(rope, position_ids, do, q, k, v):
    cos, sin = rope(v, position_ids)
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    o = flash_attn_func(q, k, v, causal=True)
    # o = o.permute(0, 2, 1, 3)
    return o

def triton_flash_fwdbwd(rope, position_ids, do, q, k, v):
    cos, sin = rope(v, position_ids)
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    o = triton_flash_attn_func(q, k, v, None, True)
    # o = o.permute(0, 2, 1, 3)
    return o


providers = [
    ("triton", "Stickbreaking", ("blue", "-")),
    ("flash", "Flash Attention", ("green", "-")),
    # ("triton_flash", "Triton Flash", ("red", "-")), # triton flash not working
]
@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=["length"],
        x_vals=[4096, 2 * 4096, 3 * 4096, 4 * 4096],
        line_arg="provider",
        line_vals=[x[0] for x in providers],
        line_names=[x[1] for x in providers],
        styles=[x[2] for x in providers],
        ylabel="ms",
        plot_name=f"triton v torch",
        args={"batch_size": 4, "num_heads": 12, "head_dim": 128, "dtype": torch.bfloat16, "bwd": True}
    )
])
def benchmark_attn(batch_size, num_heads, head_dim, length, dtype, provider, bwd):
    device = torch.device('cuda:0')
    set_seed(1337)
    warmup = 100
    rep = 1000

    q = torch.randn((batch_size, length, num_heads, head_dim), device=device, dtype=dtype)
    k = torch.randn((batch_size, length, num_heads, head_dim), device=device, dtype=dtype)
    v = torch.randn((batch_size, length, num_heads, head_dim), device=device, dtype=dtype)
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    do = torch.randn((batch_size, length, num_heads, head_dim), device=device, dtype=dtype)
    position_ids = torch.arange(q.size(1), device=device, dtype=torch.int32)[None, :]
    if provider == "triton":
        fun = lambda: tri_fwdbwd(do, q, k, v)
    elif provider == "flash":
        rope = LlamaRotaryEmbedding(dim=head_dim).to(device)
        fun = lambda: flash_fwdbwd(rope, position_ids, do, q, k, v)
    elif provider == "triton_flash":
        rope = LlamaRotaryEmbedding(dim=head_dim).to(device)
        fun = lambda: triton_flash_fwdbwd(rope, position_ids, do, q, k, v)

    if bwd:
        def fun_():
            o = fun()
            dq, dk, dv = torch.autograd.grad(o, inputs=(q, k, v), grad_outputs=do)

        return triton.testing.do_bench(fun_, warmup=warmup, rep=rep)
    else:
        return triton.testing.do_bench(fun, warmup=warmup, rep=rep)



if __name__ == "__main__":
    benchmark_attn.run(save_path=None, print_data=True)
