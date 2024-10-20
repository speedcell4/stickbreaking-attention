import torch
import pytest
import math
from torch.nn import functional as F
from stickbreaking_attention.sb_varlen import sb_flash_attn_varlen
import triton
import triton.language as tl


# for reference
def stickbreaking(q, k, v, mask, cum_weight):
    """
    Stick-breaking attention weights.
    """
    logits = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])

    original_dtype = logits.dtype

    logits = logits.float()
    log_z = F.logsigmoid(logits).masked_fill(mask, -1e5).to(original_dtype)

    log_beta = F.logsigmoid(-logits).masked_fill(mask, 0).to(original_dtype)

    re_cum_log_beta = torch.einsum('bhij,jk->bhik', log_beta, cum_weight.to(log_beta))
    log_att = log_z + re_cum_log_beta
    att = log_att.exp()
    return att @ v, 1 - att.sum(dim=-1)


def ref_fwd(q, k, v, lengths):
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

        o = o + rem[..., None] * v_chunk[None]
        outputs.append(o[0])
    return torch.cat(outputs, 1)

def ref_fwdbwd(do, q, k, v, lengths):
    o = ref_fwd(q, k, v, lengths)
    dq, dk, dv = torch.autograd.grad(o, inputs=(q, k, v), grad_outputs=do)
    return o, dq, dk, dv

def tri_fwdbwd(do, q, k, v, lengths):
    cu_seqlens = torch.cumsum(lengths, dim=-1)
    o, rem = sb_flash_attn_varlen(q, k, v, cu_seqlens,
                                  inv_temp=1 / math.sqrt(q.size(-1)),
                                  zero_start=False)
    o = o + rem[..., None] * v
    dq, dk, dv = torch.autograd.grad(o, inputs=(q, k, v), grad_outputs=do)
    return o, dq, dk, dv


providers = [
    ("reference", "Stickbreaking (ref.)", ("red", "-")),
    ("triton", "Stickbreaking", ("blue", "-")),
]
@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=["length"],
        x_vals=[4096, 8192],
        line_arg="provider",
        line_vals=[x[0] for x in providers],
        line_names=[x[1] for x in providers],
        styles=[x[2] for x in providers],
        ylabel="ms",
        plot_name=f"triton v torch",
        args={"batch_size": 1, "num_heads": 8, "head_dim": 64, "dtype": torch.bfloat16}
    )
])
def benchmark_varlen(batch_size, num_heads, head_dim, length, dtype, provider):
    device = torch.device('cuda:0')
    lengths = torch.randint(length // 2, length, (batch_size,)).to(device=device, dtype=torch.int32)
    total_length = lengths.sum()
    warmup = 100
    rep = 1000
    q = torch.randn((num_heads, total_length, head_dim), device=device, dtype=dtype)
    k = torch.randn((num_heads, total_length, head_dim), device=device, dtype=dtype)
    v = torch.randn((num_heads, total_length, head_dim), device=device, dtype=dtype)
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    do = torch.randn((num_heads, total_length, head_dim), device=device, dtype=dtype)
    if provider== "reference":
        fun = lambda: ref_fwdbwd(do, q, k, v, lengths)
    elif provider == "triton":
        fun = lambda: tri_fwdbwd(do, q, k, v, lengths)
    return triton.testing.do_bench(fun, warmup=warmup, rep=rep)



if __name__ == "__main__":
    benchmark_varlen.run(save_path=None, print_data=True)
