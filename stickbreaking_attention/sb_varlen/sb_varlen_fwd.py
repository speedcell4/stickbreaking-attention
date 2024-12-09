import math
import torch
import triton
import triton.language as tl
from . import log2, inv_log2, ALLOW_TF32
from .softplus import softplus

@triton.jit
def load_kv(KT_blk_ptrs, V_blk_ptrs,
            N_mask, NO_N_MASK,
            D_mask, NO_D_MASK: tl.constexpr):
    if NO_D_MASK:
        if NO_N_MASK:
            kT = tl.load(KT_blk_ptrs)
            v = tl.load(V_blk_ptrs)
        else:
            kT = tl.load(KT_blk_ptrs, mask=N_mask[None, :])
            v = tl.load(V_blk_ptrs, mask=N_mask[:, None])
    else:
        kT = tl.load(KT_blk_ptrs, mask=N_mask[None, :] & D_mask[:, None])
        v = tl.load(V_blk_ptrs, mask=N_mask[:, None] & D_mask[None, :])
    return kT, v


@triton.jit
def compute_block(
        q, kT, qk_scale, neg_log_acc, 
        M_blk_idxs, N_blk_idxs,
        cm, on_band,
        ALLOW_TF32: tl.constexpr,
        backward: tl.constexpr,
        use_cumsum: tl.constexpr = False):

    qk = tl.dot(q, kT, allow_tf32=ALLOW_TF32) * qk_scale

    log_om_beta = -softplus(qk) # log_om_beta (one minus beta) : log(1 - \beta)

    if on_band:
        block_mask = M_blk_idxs[:, None] > N_blk_idxs[None, :] # diagonal
        log_om_beta = tl.where(block_mask, log_om_beta, 0.)
        if backward:
            neg_log_acc -= tl.sum(log_om_beta, axis=1)
        log_p = qk + neg_log_acc[:, None]

        if use_cumsum:
            log_p += tl.cumsum(log_om_beta.to(q.dtype), axis=1, reverse=True)
        else:
            log_p = tl.dot(log_om_beta.to(q.dtype), cm, acc=log_p, allow_tf32=ALLOW_TF32)

        p = tl.math.exp2(log_p)
        p = tl.where(block_mask, p, 0.0)
    else:
        if backward:
            neg_log_acc -= tl.sum(log_om_beta, axis=1)
        log_p = qk + neg_log_acc[:, None]
        if use_cumsum:
            log_p += tl.cumsum(log_om_beta.to(q.dtype), axis=1, reverse=True)
        else:
            log_p = tl.dot(log_om_beta.to(q.dtype), cm, acc=log_p, allow_tf32=ALLOW_TF32)

        p = tl.math.exp2(log_p)
    if not backward:
        neg_log_acc += tl.sum(log_om_beta, axis=1)
    return p, log_om_beta, neg_log_acc



@triton.jit
def compute_boundaries(block_id, CSL_ptr, CPO_ptr,
                       batch_size: tl.constexpr, BLOCK_CSL: tl.constexpr, BLOCK_M: tl.constexpr):
    CSL_range = tl.arange(0, BLOCK_CSL)
    block_offsets = tl.load(CPO_ptr + CSL_range, mask=CSL_range < batch_size, other=tl.num_programs(1))
    sequence_id = tl.sum((block_id >= block_offsets).to(tl.int32), axis=0) # lookup sequence in batch
    if sequence_id == 0:
        sequence_start_offset = 0
        block_start_offset = 0
    else:
        sequence_start_offset = tl.load(CSL_ptr + sequence_id - 1).to(tl.int32)
        block_start_offset = tl.load(CPO_ptr + sequence_id - 1).to(tl.int32)
    sequence_end_offset = tl.load(CSL_ptr + sequence_id).to(tl.int32)
    block_offset = block_id - block_start_offset
    sequence_block_start_offset = sequence_start_offset + BLOCK_M * block_offset
    return sequence_start_offset, sequence_end_offset, sequence_block_start_offset, block_start_offset


def get_configs():
    return [
        triton.Config({}, num_stages=s, num_warps=w)
        for s in [4]
        for w in [4]
    ]
@triton.autotune(configs=get_configs(), key=["token_size", "head_size"])
@triton.jit
def _forward(
    Q_ptr, stride_qh, stride_qm, stride_qd,
    K_ptr, stride_kh, stride_kn, stride_kd,
    V_ptr, stride_vh, stride_vn, stride_vd,
    O_ptr, stride_oh, stride_om, stride_od,
    R_ptr, stride_rh, stride_rm,
    A_ptr, stride_ah, stride_am,
    W_ptr, stride_wh, stride_wm, stride_wn,
    CSL_ptr, CPO_ptr,
    # pid_debug_ptr,
    logit_scale: tl.constexpr,
    batch_size,
    token_size,
    head_size: tl.constexpr,
    num_heads: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_CSL: tl.constexpr,
    NO_D_MASK: tl.constexpr,
    NO_M_MASK: tl.constexpr,
    NO_N_MASK: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    inv_log2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    no_grad: tl.constexpr = False,
    acc_dtype: tl.constexpr = tl.float32,
    return_attention: tl.constexpr = False,
): 
    tl.static_assert(BLOCK_M % BLOCK_N == 0)
    head_pid = tl.program_id(0)
    prog_id = tl.program_id(1)
    # Universal stuff
    qk_scale = inv_log2 * logit_scale
    M_range = tl.arange(0, BLOCK_M)
    N_range = tl.arange(0, BLOCK_N)
    D_range = tl.arange(0, BLOCK_D)
    CSL_range = tl.arange(0, BLOCK_CSL)
    D_mask = D_range < head_size
    cm = tl.where(N_range[:, None] >= N_range[None, :], 1.0, 0.0).to(Q_ptr.type.element_ty)

    block_offsets = tl.load(CPO_ptr + CSL_range, mask=CSL_range < batch_size, other=tl.num_programs(1))
    seq_id = tl.sum((prog_id >= block_offsets).to(tl.int32), axis=0) # lookup sequence in batch
    if seq_id == 0:
        seq_start_offset = 0
        prog_id_start_offset = 0
    else:
        seq_start_offset = tl.load(CSL_ptr + seq_id - 1).to(tl.int32)
        prog_id_start_offset = tl.load(CPO_ptr + seq_id - 1).to(tl.int32)
    seq_end_offset = tl.load(CSL_ptr + seq_id).to(tl.int32)
    prog_id_end_offset = tl.load(CPO_ptr + seq_id).to(tl.int32)
    seq_length = seq_end_offset - seq_start_offset
    seq_num_progs = prog_id_end_offset - prog_id_start_offset

    # pid = tl.program_id(0) * tl.num_programs(1) + tl.program_id(1) # TODO debug
    seq_alloc_prog_id = prog_id - prog_id_start_offset
    if seq_alloc_prog_id > 0:
        # First head block
        head_id = head_pid * 2
        seq_prog_id = prog_id - prog_id_start_offset - 1
        # tl.store(pid_debug_ptr + head_id * tl.num_programs(1) + prog_id_start_offset + seq_prog_id, pid)
        Q_head_seq_ptr = Q_ptr + stride_qh * head_id + stride_qm * seq_start_offset
        K_head_seq_ptr = K_ptr + stride_kh * head_id + stride_kn * seq_start_offset
        V_head_seq_ptr = V_ptr + stride_vh * head_id + stride_vn * seq_start_offset
        O_head_seq_ptr = O_ptr + stride_oh * head_id + stride_om * seq_start_offset
        R_head_seq_ptr = R_ptr + stride_rh * head_id + stride_rm * seq_start_offset
        A_head_seq_ptr = A_ptr + stride_ah * head_id + stride_am * seq_start_offset
        W_head_seq_ptr = W_ptr + stride_wh * head_id + stride_am * seq_start_offset
        _forward_one_row(
            seq_prog_id, seq_length,
            qk_scale,
            M_range, N_range,
            D_range, D_mask, cm,
            Q_head_seq_ptr, stride_qm, stride_qd,
            K_head_seq_ptr, stride_kn, stride_kd,
            V_head_seq_ptr, stride_vn, stride_vd,
            O_head_seq_ptr, stride_om, stride_od,
            R_head_seq_ptr, stride_rm,
            A_head_seq_ptr, stride_am,
            W_head_seq_ptr, stride_wm, stride_wn,
            BLOCK_D,
            NO_D_MASK, NO_M_MASK, NO_N_MASK,
            ALLOW_TF32,
            BLOCK_M, BLOCK_N,
            no_grad, acc_dtype,
            return_attention,
        )
    if seq_num_progs - seq_alloc_prog_id - 1 > 0 and head_pid * 2 + 1 < num_heads:
        # Reverse head block
        head_id = head_pid * 2 + 1
        seq_prog_id = seq_num_progs - seq_alloc_prog_id - 1 - 1 # reverse ids
        # tl.store(pid_debug_ptr + head_id * tl.num_programs(1) + prog_id_start_offset + seq_prog_id, pid)
        Q_head_seq_ptr = Q_ptr + stride_qh * head_id + stride_qm * seq_start_offset
        K_head_seq_ptr = K_ptr + stride_kh * head_id + stride_kn * seq_start_offset
        V_head_seq_ptr = V_ptr + stride_vh * head_id + stride_vn * seq_start_offset
        O_head_seq_ptr = O_ptr + stride_oh * head_id + stride_om * seq_start_offset
        R_head_seq_ptr = R_ptr + stride_rh * head_id + stride_rm * seq_start_offset
        A_head_seq_ptr = A_ptr + stride_ah * head_id + stride_am * seq_start_offset
        W_head_seq_ptr = W_ptr + stride_wh * head_id + stride_am * seq_start_offset
        _forward_one_row(
            seq_prog_id, seq_length,
            qk_scale,
            M_range, N_range,
            D_range, D_mask, cm,
            Q_head_seq_ptr, stride_qm, stride_qd,
            K_head_seq_ptr, stride_kn, stride_kd,
            V_head_seq_ptr, stride_vn, stride_vd,
            O_head_seq_ptr, stride_om, stride_od,
            R_head_seq_ptr, stride_rm,
            A_head_seq_ptr, stride_am,
            W_head_seq_ptr, stride_wm, stride_wn,
            BLOCK_D,
            NO_D_MASK, NO_M_MASK, NO_N_MASK,
            ALLOW_TF32,
            BLOCK_M, BLOCK_N,
            no_grad, acc_dtype,
            return_attention,
        )


@triton.jit
def _forward_one_row(
    seq_prog_id, seq_length,
    qk_scale,
    M_range, N_range,
    D_range, D_mask, cm,
    Q_head_seq_ptr, stride_qm, stride_qd,
    K_head_seq_ptr, stride_kn, stride_kd,
    V_head_seq_ptr, stride_vn, stride_vd,
    O_head_seq_ptr, stride_om, stride_od,
    R_head_seq_ptr, stride_rm,
    A_head_seq_ptr, stride_am,
    W_head_seq_ptr, stride_wm, stride_wn,
    BLOCK_D: tl.constexpr,
    NO_D_MASK: tl.constexpr,
    NO_M_MASK: tl.constexpr,
    NO_N_MASK: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    no_grad: tl.constexpr = False,
    acc_dtype: tl.constexpr = tl.float32,
    return_attention: tl.constexpr = False,
):
    # Loading thread information
    block_start_offset = BLOCK_M * seq_prog_id
    M_blk_idxs = block_start_offset + M_range
    M_mask = M_blk_idxs < seq_length
    NO_M_MASK = ((block_start_offset + BLOCK_M - 1) < seq_length)

    N_blk_idxs_start = block_start_offset + BLOCK_M # BLOCK_M must be a multiple of BLOCK_N
    N_blk_idxs = N_blk_idxs_start + N_range

    # Init pointers
    Q_blk_ptrs = Q_head_seq_ptr + stride_qm * M_blk_idxs[:, None] + stride_qd * D_range[None, :]
    KT_blk_ptrs = K_head_seq_ptr + stride_kn * N_blk_idxs[None, :] + stride_kd * D_range[:, None]
    V_blk_ptrs = V_head_seq_ptr + stride_vn * N_blk_idxs[:, None] + stride_vd * D_range[None, :]
    O_blk_ptrs = O_head_seq_ptr + stride_om * M_blk_idxs[:, None] + stride_od * D_range[None, :]
    R_blk_ptrs = R_head_seq_ptr + stride_rm * M_blk_idxs
    A_blk_ptrs = A_head_seq_ptr + stride_am * M_blk_idxs

    # --- Load band vectors ---
    if NO_D_MASK:
        if NO_M_MASK:
            q = tl.load(Q_blk_ptrs)
        else:
            q = tl.load(Q_blk_ptrs, mask=M_mask[:, None], other=0.)
    else:
        q = tl.load(Q_blk_ptrs, mask=M_mask[:, None] & D_mask[None, :], other=0.)

    iters = N_blk_idxs_start // BLOCK_N
    neg_log_acc = tl.zeros([BLOCK_M], dtype=acc_dtype)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=acc_dtype)
    # --- End band vectors ---

    # Iterate only up to start of sequence
    for i in range(iters):
        N_blk_idxs -= BLOCK_N
        N_blk_idxs_start -= BLOCK_N
        KT_blk_ptrs -= BLOCK_N * stride_kn
        V_blk_ptrs -= BLOCK_N * stride_vn

        N_mask = N_blk_idxs < seq_length
        kT, v = load_kv(
            KT_blk_ptrs, V_blk_ptrs,
            N_mask=N_mask, NO_N_MASK=N_blk_idxs_start + BLOCK_N - 1 < seq_length,
            D_mask=D_mask, NO_D_MASK=NO_D_MASK
        )
        on_band = i < BLOCK_M // BLOCK_N
        p, _, neg_log_acc = compute_block(
            q, kT, qk_scale, neg_log_acc,
            M_blk_idxs, N_blk_idxs,
            cm, on_band,
            ALLOW_TF32,
            backward=False
        )
        # Store intermediate values
        acc = tl.dot(p.to(v.dtype), v, acc, allow_tf32=ALLOW_TF32)
        if return_attention: # TODO write returns_attention_weight
            tl.store(
                W_head_seq_ptr + stride_wm * M_blk_idxs[:, None] + stride_wn * N_blk_idxs[None, :],
                p,
                mask=(M_blk_idxs < sequence_end_offset)[:, None] & (N_blk_idxs < seq_length)[None, :]
            )
    if NO_M_MASK:
        tl.store(R_blk_ptrs, tl.math.exp2(neg_log_acc))
        tl.store(A_blk_ptrs, neg_log_acc.to(A_head_seq_ptr.type.element_ty))
    else:
        tl.store(R_blk_ptrs, tl.math.exp2(neg_log_acc), mask=M_mask)
        tl.store(A_blk_ptrs, neg_log_acc.to(A_head_seq_ptr.type.element_ty), mask=M_mask)
    if NO_D_MASK:
        tl.store(O_blk_ptrs, acc.to(O_head_seq_ptr.type.element_ty), mask=M_mask[:, None])
    else:
        tl.store(O_blk_ptrs, acc.to(O_head_seq_ptr.type.element_ty), mask=M_mask[:, None] & D_mask[None, :])


def calculate_programs_needed(cu_seqlens: torch.Tensor, BLOCK_SIZE):
    lens = cu_seqlens.clone()
    lens[1:] -= cu_seqlens[:-1]
    seq_num_programs = ((lens - 1) // BLOCK_SIZE) + 1 
    seq_num_programs += 1
    seq_program_offsets = torch.cumsum(seq_num_programs, dim=0)
    return seq_program_offsets

def sb_fwd(q, k, v, cu_seqlens, logit_scale=None, no_grad=False, return_attention=False):
    with torch.cuda.device(q.device):
        num_heads = q.size(0)
        batch_size = cu_seqlens.size(0)
        token_size = q.size(1)
        dim_size = q.size(-1)
        BLOCK_M = 64
        BLOCK_N = 32
        BLOCK_D = triton.next_power_of_2(dim_size)

        seq_program_offsets = calculate_programs_needed(cu_seqlens, BLOCK_SIZE=BLOCK_M)

        if logit_scale is None:
            logit_scale = 1 / math.sqrt(dim_size)

        o = torch.empty_like(q)
        rem = torch.zeros_like(q[:, :, 0], device=q.device)
        neg_log_acc = torch.zeros_like(rem, device=q.device, dtype=torch.float32)
        if return_attention:
            W = torch.full((num_heads, token_size, token_size), 0., dtype=torch.float32, device=q.device)
        else:
            W = torch.empty((1, 1, 1), device=q.device)
        num_folded_heads = triton.cdiv(num_heads, 2)
        num_seq_blocks = seq_program_offsets[-1]
        # pid_debug = torch.zeros((num_heads, num_seq_blocks), dtype=torch.int32, device=q.device)
        grid = (num_folded_heads, num_seq_blocks)
        # print(grid, math.prod(grid).item())
        _forward[grid](
            q, q.stride(0), q.stride(1), q.stride(2),
            k, k.stride(0), k.stride(1), k.stride(2),
            v, v.stride(0), v.stride(1), v.stride(2),
            o, o.stride(0), o.stride(1), o.stride(2),
            rem, rem.stride(0), rem.stride(1),
            neg_log_acc, neg_log_acc.stride(0), neg_log_acc.stride(1),
            W, W.stride(0), W.stride(1), W.stride(2),
            cu_seqlens, seq_program_offsets,
            # pid_debug,
            logit_scale=logit_scale,
            batch_size=batch_size,
            token_size=token_size,
            head_size=dim_size,
            num_heads=num_heads,
            no_grad=no_grad,
            BLOCK_D=BLOCK_D,
            BLOCK_CSL=triton.next_power_of_2(batch_size),
            NO_D_MASK=BLOCK_D == dim_size,
            NO_M_MASK=(token_size % BLOCK_M) == 0,
            NO_N_MASK=(token_size % BLOCK_N) == 0,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            ALLOW_TF32=ALLOW_TF32,
            inv_log2=inv_log2,
            return_attention=return_attention,
        )
        # print(pid_debug)
        if return_attention:
            return o, rem, neg_log_acc, seq_program_offsets, W
        else:
            return o, rem, neg_log_acc, seq_program_offsets


