import math

import torch
import triton
from flash_attn import flash_attn_varlen_func
from torch.nn import functional as F
from transformers import set_seed
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, rotate_half

from stickbreaking_attention.sb_ref import stickbreaking
from stickbreaking_attention.sb_varlen import sb_attn_varlen


def ref_fwd(q, k, v, lengths):
    q = q.permute(1, 0, 2)
    k = k.permute(1, 0, 2)
    v = v.permute(1, 0, 2)
    splits = list(lengths.cpu().numpy())
    max_len = max(splits)
    cm = torch.ones(max_len, max_len).tril(-1).to(q)
    mask = torch.ones(max_len, max_len).triu(0).cuda().bool()
    outputs = []
    for q_chunk, k_chunk, v_chunk in zip(q.split(splits, 1), k.split(splits, 1), v.split(splits, 1)):
        len = q_chunk.size(1)
        o, rem = stickbreaking(
            q_chunk[None, :],
            k_chunk[None, :],
            v_chunk[None, :],
            mask[:len, :len], cm[:len, :len]
        )

        # o = o + rem[..., None] * v_chunk[None]
        outputs.append(o[0])
    return torch.cat(outputs, 1)


def ref_fwdbwd(do, q, k, v, lengths):
    o = ref_fwd(q, k, v, lengths)
    return o


def tri_fwdbwd(do, q, k, v, lengths):
    q = q.permute(1, 0, 2)
    k = k.permute(1, 0, 2)
    v = v.permute(1, 0, 2)
    cu_seqlens = torch.cumsum(lengths, dim=-1)
    o, rem = sb_attn_varlen(q, k, v,
                            cu_seqlens=cu_seqlens,
                            max_seqlens=max(lengths).item(),
                            inv_temp=1 / math.sqrt(q.size(-1)),
                            zero_start=False)
    # o = o + rem[..., None] * v
    return o


def flash_fwdbwd(rope, position_ids, do, q, k, v, lengths):
    cos, sin = rope(v, position_ids)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    lengths = lengths.to(torch.int32)
    cu_seqlens = torch.cumsum(lengths, dim=-1)
    cu_seqlens = F.pad(cu_seqlens, (1, 0)).to(torch.int32)
    max_len = torch.max(lengths)
    o = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_len,
        max_seqlen_k=max_len,
        causal=True
    )
    o = o.permute(1, 0, 2)
    return o


providers = [
    # ("reference", "Stickbreaking (ref.)", ("red", "-")),
    ("triton", "Stickbreaking", ("blue", "-")),
    ("flash", "Flash Attention", ("green", "-")),
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
def benchmark_varlen(batch_size, num_heads, head_dim, length, dtype, provider, bwd):
    device = torch.device('cuda:0')
    set_seed(1337)
    lengths = torch.randint(length, length + 1, (batch_size,)).to(device=device, dtype=torch.int32)
    total_length = lengths.sum()
    warmup = 100
    rep = 1000

    q = torch.randn((total_length, num_heads, head_dim), device=device, dtype=dtype)
    k = torch.randn((total_length, num_heads, head_dim), device=device, dtype=dtype)
    v = torch.randn((total_length, num_heads, head_dim), device=device, dtype=dtype)
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    do = torch.randn((num_heads, total_length, head_dim), device=device, dtype=dtype)
    position_ids = torch.arange(q.size(1), device=device, dtype=torch.int32)[None, :]

    if provider == "reference":
        fun = lambda: ref_fwdbwd(do, q, k, v, lengths)
    elif provider == "triton":
        fun = lambda: tri_fwdbwd(do, q, k, v, lengths)
    elif provider == "flash":
        config = LlamaConfig(max_position_embeddings=length)
        rope = LlamaRotaryEmbedding(config).to(device)
        fun = lambda: flash_fwdbwd(rope, position_ids, do, q, k, v, lengths)
    if bwd:
        def fun_():
            o = fun()
            dq, dk, dv = torch.autograd.grad(o, inputs=(q, k, v), grad_outputs=do)

        return triton.testing.do_bench(fun_, warmup=warmup, rep=rep)
    else:
        return triton.testing.do_bench(fun, warmup=warmup, rep=rep)


if __name__ == "__main__":
    benchmark_varlen.run(save_path=None, print_data=True)
