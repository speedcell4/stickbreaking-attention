import math
import torch
import triton
import triton.language as tl
from torch.nn import functional as F


log2 = math.log(2)
inv_log2 = 1 / log2
ALLOW_TF32 = False
# WARNING: does not work when bfloat16 and 64. Triton problem.
BLOCK_M = 32
BLOCK_N = 32

@triton.jit
def softplus(x):
    out = tl.where(x < 15., tl.math.log2(1 + tl.math.exp2(x)), x)
    # out = tl.maximum(0, x)
    return out


@triton.jit
def get_batch_ids(CSL_ptr, batch_size: tl.constexpr, token_size, M_blk_idxs, CSL_BLOCK: tl.constexpr=4):
    CSL_range = tl.arange(0, CSL_BLOCK)
    cuseqlens = tl.load(CSL_ptr + CSL_range, mask=CSL_range < batch_size, other=token_size).to(tl.int32)
    batch_ids = tl.sum(tl.where(M_blk_idxs[:, None] >= cuseqlens[None, :], 1., 0.), axis=1).to(tl.int32)
    return batch_ids

@triton.jit
def load_kv(K_blk_ptrs, V_blk_ptrs, D_mask, N_mask,
            NO_D_MASK: tl.constexpr, NO_N_MASK: tl.constexpr, NO_N_MASK_):
    if NO_D_MASK:
        if NO_N_MASK or NO_N_MASK_:
            k = tl.load(K_blk_ptrs)
            v = tl.load(V_blk_ptrs)
        else:
            k = tl.load(K_blk_ptrs, mask=N_mask[:, None])
            v = tl.load(V_blk_ptrs, mask=N_mask[:, None])
    else:
        ND_mask = N_mask[:, None] & D_mask[None, :]
        k = tl.load(K_blk_ptrs, mask=ND_mask)
        v = tl.load(V_blk_ptrs, mask=ND_mask)
    return k,v

@triton.jit
def locked_add(Lock_ptr, Count_ptr, A_ptrs, a, B_ptrs, b, mask, NO_MASK_, NO_MASK: tl.constexpr):
    # tl.static_print(Lock_ptr)
    locked = tl.atomic_cas(Lock_ptr, 0, 1)
    while locked == 1:
        locked = tl.atomic_cas(Lock_ptr, 0, 1)

    count = tl.load(Count_ptr)
    if NO_MASK or NO_MASK_:
        if count == 0:
            tl.store(A_ptrs, a)
            tl.store(B_ptrs, b)
            tl.store(Count_ptr, count + 1)
        else:
            tl.store(A_ptrs, a + tl.load(A_ptrs))
            tl.store(B_ptrs, b + tl.load(B_ptrs))
    else:
        if count == 0:
            tl.store(A_ptrs, a, mask=mask)
            tl.store(B_ptrs, b, mask=mask)
            tl.store(Count_ptr, count + 1)
        else:
            tl.store(A_ptrs, a + tl.load(A_ptrs, mask=mask), mask=mask)
            tl.store(B_ptrs, b + tl.load(B_ptrs, mask=mask), mask=mask)

    tl.atomic_xchg(Lock_ptr, 0)

@triton.jit
def compute_block(
        q, k, qk_scale, neg_log_acc, 
        M_blk_idxs, N_blk_idxs, start_idxs, 
        cm, on_band,
        needs_mask,
        ALLOW_TF32,
        backward):
    qk = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32) * qk_scale

    log_om_beta = -softplus(qk) # log_om_beta (one minus beta) : log(1 - \beta)

    if needs_mask:
        block_mask = N_blk_idxs[None, :] >= start_idxs[:, None]
        if on_band:
            block_mask &= M_blk_idxs[:, None] > N_blk_idxs[None, :] # diagonal
        log_om_beta = tl.where(block_mask, log_om_beta, 0.)
        if backward:
            neg_log_acc -= tl.sum(log_om_beta, axis=1)
        log_p = qk + tl.dot(log_om_beta.to(q.dtype), cm, allow_tf32=ALLOW_TF32) + neg_log_acc[:, None]
        p = tl.math.exp2(log_p)
        p = tl.where(block_mask, p, 0.0)
    else:
        
        if backward:
            neg_log_acc -= tl.sum(log_om_beta, axis=1)
        log_p = qk + tl.dot(log_om_beta.to(q.dtype), cm, allow_tf32=ALLOW_TF32) + neg_log_acc[:, None]
        p = tl.math.exp2(log_p)
    if not backward:
        neg_log_acc += tl.sum(log_om_beta, axis=1)
    return p, log_om_beta, neg_log_acc


# @triton.autotune(configs=[triton.Config({}, num_stages=4, num_warps=4)], key=[],)

def get_configs():
    if True:
        return [triton.Config({"GROUP_SIZE": 1}, num_warps=4, num_stages=4)]
    else:
        return [
            triton.Config({"GROUP_SIZE": g}, num_stages=s, num_warps=w)
            for s in [1, 3, 4, 7]
            for w in [1, 2, 4, 8]
            for g in [1, 2, 4, 8, 16, 32, 64]
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
    logit_scale,
    batch_size,
    token_size,
    head_size: tl.constexpr,
    num_heads: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_CSL: tl.constexpr,
    NO_D_MASK: tl.constexpr,
    NO_M_MASK: tl.constexpr,
    NO_N_MASK: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    inv_log2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    no_grad: tl.constexpr = False,
    acc_dtype: tl.constexpr = tl.float32,
): 
    head_id = tl.program_id(0)
    block_id = tl.program_id(1)

    CSL_range = tl.arange(0, BLOCK_CSL)
    block_offsets = tl.load(CPO_ptr + CSL_range, mask=CSL_range < batch_size, other=tl.num_programs(1))
    sequence_id = tl.sum((block_id > block_offsets).to(tl.int32), axis=0)
    if sequence_id == 0:
        sequence_start_offset = 0
    else:
        sequence_start_offset = tl.load(CSL_ptr + sequence_id - 1).to(tl.int32)
    sequence_block_offset = block_id - tl.load(CPO_ptr + sequence_id)
    sequence_offset = sequence_start_offset + BLOCK_M * sequence_block_offset

    qk_scale = inv_log2 * logit_scale
    M_range = tl.arange(0, BLOCK_M)
    N_range = tl.arange(0, BLOCK_N)
    D_range = tl.arange(0, BLOCK_D)

    D_mask = D_range < head_size

    cm = tl.where(N_range[:, None] >= N_range[None, :], 1.0, 0.0).to(Q_ptr.type.element_ty)

    _forward_one_row(
        M_block_id, head_id,
        qk_scale,
        M_range, N_range,
        D_range, D_mask, cm,
        Q_ptr, stride_qh, stride_qm, stride_qd,
        K_ptr, stride_kh, stride_kn, stride_kd,
        V_ptr, stride_vh, stride_vn, stride_vd,
        O_ptr, stride_oh, stride_om, stride_od,
        R_ptr, stride_rh, stride_rm,
        A_ptr, stride_ah, stride_am,
        W_ptr, stride_wh, stride_wm, stride_wn,
        CSL_ptr,
        batch_size,
        token_size,
        BLOCK_D, BLOCK_CSL,
        NO_D_MASK, NO_M_MASK, NO_N_MASK,
        ALLOW_TF32,
        BLOCK_M, BLOCK_N,
        no_grad, acc_dtype,
    )

@triton.jit
def _forward_one_row(
    M_block_id, head_id,
    qk_scale,
    M_range,
    N_range,
    D_range, D_mask, cm,
    Q_ptr, stride_qh, stride_qm, stride_qd,
    K_ptr, stride_kh, stride_kn, stride_kd,
    V_ptr, stride_vh, stride_vn, stride_vd,
    O_ptr, stride_oh, stride_om, stride_od,
    R_ptr, stride_rh, stride_rm,
    A_ptr, stride_ah, stride_am,
    W_ptr, stride_wh, stride_wm, stride_wn,
    CSL_ptr,
    batch_size,
    token_size,
    BLOCK_D: tl.constexpr,
    BLOCK_CSL: tl.constexpr,
    NO_D_MASK: tl.constexpr,
    NO_M_MASK: tl.constexpr,
    NO_N_MASK: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    no_grad: tl.constexpr = False,
    acc_dtype: tl.constexpr = tl.float32,
): 

    # Loading thread information
    M_blk_idxs = M_block_id * BLOCK_M + M_range, BLOCK_M
    M_mask = M_blk_idxs < token_size

    NO_M_MASK = NO_M_MASK or (((M_block_id + 1) * BLOCK_M - 1) < token_size)
    end_m = (M_block_id + 1) * BLOCK_M
    N_block_id = end_m
    N_blk_idxs = N_block_id + N_range
    last_N_block_id = end_m // BLOCK_N

    # Init pointers
    Q_blk_ptrs = Q_ptr + stride_qh * head_id + stride_qm * M_blk_idxs[:, None] + stride_qd * D_range[None, :]
    K_blk_ptrs = K_ptr + stride_kh * head_id + stride_kn * N_blk_idxs[:, None] + stride_kd * D_range[None, :]
    V_blk_ptrs = V_ptr + stride_vh * head_id + stride_vn * N_blk_idxs[:, None] + stride_vd * D_range[None, :]
    O_blk_ptrs = O_ptr + stride_oh * head_id + stride_om * M_blk_idxs[:, None] + stride_od * D_range[None, :]

    R_blk_ptrs = R_ptr + stride_rh * head_id + stride_rm * M_blk_idxs
    A_blk_ptrs = A_ptr + stride_ah * head_id + stride_am * M_blk_idxs

    # --- Load band vectors ---
    if NO_D_MASK:
        if NO_M_MASK:
            q = tl.load(Q_blk_ptrs)
        else:
            q = tl.load(Q_blk_ptrs, mask=M_mask[:, None], other=0.)
    else:
        q = tl.load(Q_blk_ptrs, mask=M_mask[:, None] & D_mask[None, :], other=0.)
    # q = q.to(acc_dtype)

    M_batch_ids = get_batch_ids(CSL_ptr, batch_size, token_size, M_blk_idxs, BLOCK_CSL)
    start_idxs = tl.load(CSL_ptr + M_batch_ids - 1, mask=M_batch_ids > 0, other=0)
    # start_idxs = tl.where(M_mask, start_idxs, token_size)
    first_N_block_id = tl.min(start_idxs) // BLOCK_N
    iters = last_N_block_id - first_N_block_id

    neg_log_acc = tl.zeros([BLOCK_M], dtype=acc_dtype)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=acc_dtype)
    # --- End band vectors ---
    # Iterate only up to start of sequence
    min_start_idxs = tl.min(start_idxs)
    same_seq = min_start_idxs == tl.max(start_idxs)
    for i in range(iters):
        N_block_id -= 1
        N_blk_idxs -= BLOCK_N
        K_blk_ptrs -= BLOCK_N * stride_kn
        V_blk_ptrs -= BLOCK_N * stride_vn
        NO_N_MASK_ = ((N_block_id + 1) * BLOCK_N - 1) < token_size
        N_mask = N_blk_idxs < token_size

        k, v = load_kv(K_blk_ptrs, V_blk_ptrs, D_mask, N_mask, NO_D_MASK, NO_N_MASK, NO_N_MASK_)

        on_band = i < BLOCK_M // BLOCK_N
        is_left_edge = i == iters - 1
        needs_mask = (not same_seq) or (on_band or is_left_edge)

        p, _, neg_log_acc = compute_block(
            q, k, qk_scale, neg_log_acc,
            M_blk_idxs, N_blk_idxs, start_idxs,
            cm, on_band, needs_mask, ALLOW_TF32,
            backward=False
        )
        # Store intermediate values
        acc = tl.dot(p.to(v.dtype), v, acc, allow_tf32=ALLOW_TF32)
 
        if False: # TODO write returns_attention_weight
            tl.store(
                W_ptr + stride_wh * head_id + stride_wm * M_blk_idxs[:, None] + stride_wn * N_blk_idxs[None, :],
                p, # block_mask.to(tl.float32),
                mask=(M_blk_idxs < token_size)[:, None] & (N_blk_idxs < token_size)[None, :]
            )
    tl.store(O_blk_ptrs, acc.to(O_ptr.type.element_ty), mask=M_mask[:, None] & D_mask[None, :])
    tl.store(R_blk_ptrs, tl.math.exp2(neg_log_acc), mask=M_mask)
    tl.store(A_blk_ptrs, neg_log_acc.to(A_ptr.type.element_ty), mask=M_mask)

def calculate_programs_needed(cu_seqlens: torch.Tensor):
    lens = cu_seqlens.clone()
    lens[1:] -= cu_seqlens[:-1]
    seq_num_programs = ((lens - 1) // BLOCK_M) + 1 
    seq_program_offsets = torch.cumsum(seq_num_programs, dim=0)
    return seq_program_offsets



def sb_fwd(q, k, v, cu_seqlens, logit_scale=None, no_grad=False):
    with torch.cuda.device(q.device):
        num_heads = q.size(0)
        batch_size = cu_seqlens.size(0)
        token_size = q.size(1)
        dim_size = q.size(-1)
        BLOCK_D = triton.next_power_of_2(dim_size)

        if logit_scale is None:
            logit_scale = 1 / math.sqrt(dim_size)

        o = torch.zeros_like(q)
        rem = torch.zeros_like(q[:, :, 0], device=q.device)
        neg_log_acc = torch.zeros_like(rem, device=q.device, dtype=torch.float32)


        if False:
            W = torch.full((num_heads, token_size, token_size), 0., dtype=torch.float32, device=q.device)
        else:
            W = torch.empty((1, 1, 1), device=q.device)

        seq_program_offsets = calculate_programs_needed(cu_seqlens)
        grid = (num_heads, seq_program_offsets[-1])
        _forward[grid](
            q, q.stride(0), q.stride(1), q.stride(2),
            k, k.stride(0), k.stride(1), k.stride(2),
            v, v.stride(0), v.stride(1), v.stride(2),
            o, o.stride(0), o.stride(1), o.stride(2),
            rem, rem.stride(0), rem.stride(1),
            neg_log_acc, neg_log_acc.stride(0), neg_log_acc.stride(1),
            W, W.stride(0), W.stride(1), W.stride(2),
            cu_seqlens, seq_program_offsets,
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
            inv_log2=inv_log2
        )
        return o, rem, neg_log_acc


def init_to_zero(names):
    def fun(nargs):
        for n in names:
            nargs[n].zero_()
    return fun
    
@triton.autotune(configs=get_configs(), key=["token_size", "head_size"], 
                 reset_to_zero=["DK_ptr", "DV_ptr"])
@triton.jit
def _backward(
    DO_ptr, stride_doh, stride_dom, stride_dod,
    DR_ptr, stride_drh, stride_drm,
    A_ptr, stride_ah, stride_am,
    Q_ptr, stride_qh, stride_qm, stride_qd,
    K_ptr, stride_kh, stride_kn, stride_kd,
    V_ptr, stride_vh, stride_vn, stride_vd,
    DQ_ptr, stride_dqh, stride_dqm, stride_dqd,
    DK_ptr, stride_dkh, stride_dkn, stride_dkd,
    DV_ptr, stride_dvh, stride_dvn, stride_dvd,
    KV_Lock_ptr, KV_Count_ptr, stride_kvl,
    CSL_ptr,
    logit_scale,
    batch_size,
    token_size,
    head_size: tl.constexpr,
    num_heads: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_CSL: tl.constexpr,
    NO_D_MASK: tl.constexpr,
    NO_M_MASK: tl.constexpr,
    NO_N_MASK: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    inv_log2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    acc_dtype: tl.constexpr = tl.float32,
):
    pid = tl.program_id(0) 
    num_pids = tl.num_programs(0) 
    M_block_count = num_pids // num_heads
    head_id = pid % num_heads
    rev_M_block_id = pid // num_heads
    head_id, rev_M_block_id = tl.swizzle2d(head_id, rev_M_block_id, num_heads, M_block_count, GROUP_SIZE)

    M_block_id = M_block_count - rev_M_block_id - 1
    qk_scale = inv_log2 * logit_scale
    M_range = tl.arange(0, BLOCK_M)
    N_range = tl.arange(0, BLOCK_N)
    D_range = tl.arange(0, BLOCK_D)
    D_mask = D_range < head_size
    cm = tl.where(N_range[:, None] >= N_range[None, :], 1.0, 0.0).to(Q_ptr.type.element_ty)

    # Start by calculating the output block

    M_blk_idxs = M_block_id * BLOCK_M + M_range
    NO_M_MASK = NO_M_MASK or (M_block_id + 1) * BLOCK_M - 1 < token_size
    M_mask = M_blk_idxs < token_size
    # Load all sequence boundaries
    batch_ids = get_batch_ids(CSL_ptr, batch_size, token_size, M_blk_idxs, BLOCK_CSL)
    # Loading important thread information
    start_idxs = tl.load(CSL_ptr + batch_ids - 1, mask=batch_ids > 0, other=0) # .to(tl.int32)

    # M_start_idx = tl.load(CRB_ptr + M_block_id - 1, mask=M_block_id > 0, other=0)
    end_m = (M_block_id + 1) * BLOCK_M
    last_N_block_id = end_m // BLOCK_N
    first_N_block_id = tl.min(start_idxs) // BLOCK_N
    N_blk_idxs = first_N_block_id * BLOCK_N + N_range

    # Init pointers
    # Inputs
    Q_blk_ptrs = Q_ptr + stride_qh * head_id + stride_qm * M_blk_idxs[:, None] + stride_qd * D_range[None, :]
    K_blk_ptrs = K_ptr + stride_kh * head_id + stride_kn * N_blk_idxs[:, None] + stride_kd * D_range[None, :]
    V_blk_ptrs = V_ptr + stride_vh * head_id + stride_vn * N_blk_idxs[:, None] + stride_vd * D_range[None, :]
    DO_blk_ptrs = DO_ptr + stride_doh * head_id + stride_dom * M_blk_idxs[:, None] + stride_dod * D_range[None, :]
    A_blk_ptrs = A_ptr + stride_ah * head_id + stride_am * M_blk_idxs
    # Outputs
    DQ_blk_ptrs = DQ_ptr + stride_dqh * head_id + stride_dqm * M_blk_idxs[:, None] + stride_dqd * D_range[None, :]
    DK_blk_ptrs = DK_ptr + stride_dkh * head_id + stride_dkn * N_blk_idxs[:, None] + stride_dkd * D_range[None, :]
    DV_blk_ptrs = DV_ptr + stride_dvh * head_id + stride_dvn * N_blk_idxs[:, None] + stride_dvd * D_range[None, :]
    DR_blk_ptrs = DR_ptr + stride_drh * head_id + stride_drm * M_blk_idxs

    # --- Load band vectors ---
    if NO_D_MASK:
        if NO_M_MASK:
            q = tl.load(Q_blk_ptrs)
            do = tl.load(DO_blk_ptrs)
            dr = tl.load(DR_blk_ptrs)
            neg_log_acc = tl.load(A_blk_ptrs, mask=M_mask)
        else:
            q = tl.load(Q_blk_ptrs, mask=M_mask[:, None])
            do = tl.load(DO_blk_ptrs, mask=M_mask[:, None])
            dr = tl.load(DR_blk_ptrs, mask=M_mask)
            neg_log_acc = tl.load(A_blk_ptrs, mask=M_mask)
    else:
        MD_mask = M_mask[:, None] & D_mask[None, :]
        q = tl.load(Q_blk_ptrs, mask=MD_mask)
        do = tl.load(DO_blk_ptrs, mask=MD_mask)
        dr = tl.load(DR_blk_ptrs, mask=M_mask)
        neg_log_acc = tl.load(A_blk_ptrs, mask=M_mask)

    # Init accumulators
    neg_log_acc = neg_log_acc.to(dtype=acc_dtype)
    grad_prev_acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    # --- End band vectors ---

    # Iterate only up to start of sequence
    min_start_idxs = tl.min(start_idxs)
    same_seq = min_start_idxs == tl.max(start_idxs)
    iters = last_N_block_id - first_N_block_id
    N_block_id = first_N_block_id

    for i in range(iters):
        on_band = (iters - i - 1) < BLOCK_M // BLOCK_N
        is_left_edge = i == 0
        needs_mask = (not same_seq) or (on_band or is_left_edge)
        N_mask = N_blk_idxs < token_size

        NO_N_MASK_ = ((N_block_id + 1) * BLOCK_N - 1) < token_size

        # --- Recompute block ---
        k, v = load_kv(K_blk_ptrs, V_blk_ptrs, D_mask, N_mask, NO_D_MASK, NO_N_MASK, NO_N_MASK_)
        p, log_om_beta, neg_log_acc = compute_block( # TODO make sure the backward does correct jumps
            q, k, qk_scale, neg_log_acc,
            M_blk_idxs, N_blk_idxs, start_idxs,
            cm, on_band, needs_mask, ALLOW_TF32,
            backward=True
        )
        if not NO_M_MASK:
            neg_log_acc = tl.where(M_mask, neg_log_acc, 0.)

        # --- Do gradient stuff ---
        if tl.max(p) > 0:

            dA = tl.dot(do, tl.trans(v), allow_tf32=ALLOW_TF32) - dr[:, None]
            att_dA = (p * dA).to(cm.dtype)
            cumul_att_dA = tl.dot(att_dA, tl.trans(cm), allow_tf32=ALLOW_TF32) + grad_prev_acc[:, None]
            grad_prev_acc += tl.sum(att_dA, axis=1)
            beta = (1 - tl.exp2(log_om_beta.to(tl.float32)))
            dqk = att_dA - beta * cumul_att_dA
            # tl.static_print("dqk", dqk)

            block_dk = tl.dot(tl.trans(dqk), q.to(dqk.dtype), allow_tf32=ALLOW_TF32) * logit_scale
            block_dv = tl.dot(tl.trans(p), do.to(p.dtype), allow_tf32=ALLOW_TF32)
            locked_add(
                KV_Lock_ptr + stride_kvl * head_id + N_block_id,
                KV_Count_ptr + stride_kvl * head_id + N_block_id,
                DK_blk_ptrs, block_dk,
                DV_blk_ptrs, block_dv,
                mask=N_mask[:, None] & D_mask[None, :],
                NO_MASK_=NO_N_MASK_ and NO_D_MASK,
                NO_MASK=NO_N_MASK and NO_D_MASK # static masking
            )

            dq += tl.dot(dqk.to(k.dtype), k, allow_tf32=ALLOW_TF32)
        # --- End gradient stuff ---

        N_block_id += 1
        N_blk_idxs += BLOCK_N
        K_blk_ptrs += BLOCK_N * stride_kn
        V_blk_ptrs += BLOCK_N * stride_vn
        DK_blk_ptrs += BLOCK_N * stride_dkn
        DV_blk_ptrs += BLOCK_N * stride_dvn

    tl.store(DQ_blk_ptrs, (logit_scale * dq).to(DQ_ptr.type.element_ty), mask=M_mask[:, None] & D_mask[None, :])



def sb_bwd(do, dr, q, k, v, cu_seqlens, neg_log_acc, logit_scale=None):
    with torch.cuda.device(q.device):
        batch_size = cu_seqlens.size(0)
        num_heads = q.size(0)
        token_size = q.size(1)
        dim_size = q.size(-1)
        if logit_scale is None:
            logit_scale = 1 / math.sqrt(dim_size)
        BLOCK_D = triton.next_power_of_2(dim_size)
        # BLOCK_BATCH = triton.next_power_of_2(batch_size)
        M_count = triton.cdiv(token_size, BLOCK_M)
        N_count = triton.cdiv(token_size, BLOCK_N)

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        dkdv_lock = torch.zeros((num_heads, M_count), dtype=torch.int32, device=q.device)
        dkdv_count = torch.zeros((num_heads, M_count), dtype=torch.int32, device=q.device)

        _backward[(num_heads * M_count,)](
            # DO_ptr, stride_doh, stride_dom, stride_dod,
            do, do.stride(0), do.stride(1), do.stride(2),
            # DR_ptr, stride_drh, stride_drm,
            dr, dr.stride(0), dr.stride(1),
            # A_ptr, stride_ah, stride_am,
            neg_log_acc, neg_log_acc.stride(0), neg_log_acc.stride(1),
            # Q_ptr, stride_qh, stride_qm, stride_qd,
            q, q.stride(0), q.stride(1), q.stride(2),
            # K_ptr, stride_kh, stride_kn, stride_kd,
            k, k.stride(0), k.stride(1), k.stride(2),
            # V_ptr, stride_vh, stride_vn, stride_vd,
            v, v.stride(0), v.stride(1), v.stride(2),
            # DQ_ptr, stride_dqh, stride_dqm, stride_dqd,
            dq, dq.stride(0), dq.stride(1), dq.stride(2),
            # DK_ptr, stride_dkh, stride_dkn, stride_dkd,
            dk, dk.stride(0), dk.stride(1), dk.stride(2),
            # DV_ptr, stride_dvh, stride_dvn, stride_dvd,
            dv, dv.stride(0), dv.stride(1), dv.stride(2),
            # KV_Lock_ptr, KV_Count_ptr, stride_kvl,
            dkdv_lock, dkdv_count, dkdv_lock.stride(0),
            cu_seqlens, 
            logit_scale=logit_scale,
            batch_size=batch_size,
            token_size=token_size,
            head_size=dim_size,
            num_heads=num_heads,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
            BLOCK_CSL=triton.next_power_of_2(batch_size),
            NO_D_MASK=BLOCK_D == dim_size,
            NO_M_MASK=(token_size % BLOCK_M) == 0,
            NO_N_MASK=(token_size % BLOCK_N) == 0,
            ALLOW_TF32=ALLOW_TF32,
            inv_log2=inv_log2
        )
        # print("dkdv_count")
        # print(dkdv_count[0])
        return dq, dk, dv


class StickBreakingAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens, inv_temp):
        no_grad = not ctx.needs_input_grad[0]
        logit_scale = inv_temp
        o, rem, neg_log_acc = sb_fwd(
            q, k, v, cu_seqlens,
            logit_scale=inv_temp,
            no_grad=no_grad
        )
        ctx.save_for_backward(q, k, v, neg_log_acc, cu_seqlens)
        ctx.logit_scale = logit_scale
        return o, rem

    @staticmethod
    def backward(ctx, do, drem):
        logit_scale = ctx.logit_scale
        q, k, v, neg_log_acc, cu_seqlens = ctx.saved_tensors
        dq, dk, dv = sb_bwd(
            do, drem, q, k, v, cu_seqlens, neg_log_acc, logit_scale
        )
        return dq, dk, dv, None, None


def sb_attn_varlen(q, k, v, cu_seqlens, inv_temp=None, zero_start=True):
    if zero_start:
        assert cu_seqlens[0] == 0
        cu_seqlens = cu_seqlens[1:]
    if inv_temp is None:
        inv_temp = 1 / math.sqrt(q.size(-1))
    # with torch.no_grad():
    #     cu_row_blocks, first_row_block, sequence_ids = row_block_counts_and_sequence_ids(cu_seqlens, BLOCK_M, BLOCK_N)

    return sb_attn_varlen_(q, k, v, inv_temp, cu_seqlens)


def sb_attn_varlen_(q, k, v, inv_temp, cu_seqlens):
    return StickBreakingAttention.apply(q, k, v, cu_seqlens, inv_temp)
