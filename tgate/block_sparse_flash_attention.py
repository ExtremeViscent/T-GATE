import torch
import numpy as np

import triton
import triton.language as tl

from flash_attn import flash_attn_func


# @triton.autotune(
#    configs=[
#        triton.Config({}, num_stages=1, num_warps=4),
#        triton.Config({}, num_stages=1, num_warps=8),
#        triton.Config({}, num_stages=2, num_warps=4),
#        triton.Config({}, num_stages=2, num_warps=8),
#        triton.Config({}, num_stages=3, num_warps=4),
#        triton.Config({}, num_stages=3, num_warps=8),
#        triton.Config({}, num_stages=4, num_warps=4),
#        triton.Config({}, num_stages=4, num_warps=8),
#        triton.Config({}, num_stages=5, num_warps=4),
#        triton.Config({}, num_stages=5, num_warps=8),
#    ],
#    key=[],
# )
@triton.jit
def triton_block_sparse_attn_kernel_v1(
    Q, K, V, sm_scale,
    block_count, block_index,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    num_rows, num_cols,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    dtype: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh

    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    num_blks = tl.load(block_count + off_hz * num_rows + start_m)
    blks_ptr = block_index + (off_hz * num_rows + start_m) * num_cols

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504

    m_mask = offs_m[:, None] < N_CTX

    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs, mask=m_mask)
    q = (q * qk_scale).to(dtype)

    # loop over k, v and update accumulator
    for block_index in range(num_blks - 1):
        cols = tl.load(blks_ptr + block_index) * BLOCK_N + offs_n
        # -- load k, v --
        k = tl.load(k_ptrs + cols[None, :] * stride_kn)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn)
        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        # -- compute scaling constant --
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    cols = tl.load(blks_ptr + num_blks - 1) * BLOCK_N + offs_n
    n_mask = cols < N_CTX
    # -- load k, v --
    k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :])
    v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None])
    # -- compute qk --
    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk = tl.where(n_mask[None, :], qk, float("-inf"))
    qk += tl.dot(q, k)
    # -- compute scaling constant --
    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    # -- scale and update acc --
    acc_scale = l_i * 0 + alpha  # workaround some compiler bug
    acc *= acc_scale[:, None]
    acc += tl.dot(p.to(dtype), v)
    # -- update m_i and l_i --
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new

    # write back O
    acc /= l_i[:, None]
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask)


# @triton.autotune(
#    configs=[
#        triton.Config({}, num_stages=1, num_warps=4),
#        triton.Config({}, num_stages=1, num_warps=8),
#        triton.Config({}, num_stages=2, num_warps=4),
#        triton.Config({}, num_stages=2, num_warps=8),
#        triton.Config({}, num_stages=3, num_warps=4),
#        triton.Config({}, num_stages=3, num_warps=8),
#        triton.Config({}, num_stages=4, num_warps=4),
#        triton.Config({}, num_stages=4, num_warps=8),
#        triton.Config({}, num_stages=5, num_warps=4),
#        triton.Config({}, num_stages=5, num_warps=8),
#    ],
#    key=[],
# )
@triton.jit
def triton_block_sparse_attn_kernel_v2(
    Q, K, V, sm_scale,
    block_count, block_index,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    num_rows, num_cols,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    dtype: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qo_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    num_blks = tl.load(block_count + off_hz * num_rows + start_m)
    blks_ptr = block_index + (off_hz * num_rows + start_m) * num_cols
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr, boundary_check=(1, 0), padding_option='zero')
    q = (q * qk_scale).to(dtype)

    # loop over k, v and update accumulator
    start_n = 0

    for block_index in range(num_blks - 1):
        stride = tl.load(blks_ptr + block_index) * BLOCK_N

        # update pointers
        start_n += stride
        K_block_ptr = tl.advance(K_block_ptr, (0, stride))
        V_block_ptr = tl.advance(V_block_ptr, (stride, 0))

        # -- load k, v --
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        # -- compute scaling constant --
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    stride = tl.load(blks_ptr + num_blks - 1) * BLOCK_N

    # update pointers
    start_n += stride
    K_block_ptr = tl.advance(K_block_ptr, (0, stride))
    V_block_ptr = tl.advance(V_block_ptr, (stride, 0))

    n_mask = start_n + offs_n[None, :] < N_CTX
    # -- load k, v --
    k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option='zero')
    v = tl.load(V_block_ptr, boundary_check=(1, 0), padding_option='zero')
    # -- compute qk --
    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk = tl.where(n_mask, qk, float("-inf"))
    qk += tl.dot(q, k)
    # -- compute scaling constant --
    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    # -- scale and update acc --
    acc_scale = l_i * 0 + alpha  # workaround some compiler bug
    acc *= acc_scale[:, None]
    acc += tl.dot(p.to(dtype), v)
    # -- update m_i and l_i --
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new

    # write back O
    acc /= l_i[:, None]
    O_block_ptr = tl.make_block_ptr(
        base=Out + qo_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_ok),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(dtype), boundary_check=(1, 0))


def triton_block_sparse_forward(
    q: torch.Tensor,            # [BATCH, N_HEADS, N_CTX, D_HEAD]
    k: torch.Tensor,            # [BATCH, N_HEADS, N_CTX, D_HEAD]
    v: torch.Tensor,            # [BATCH, N_HEADS, N_CTX, D_HEAD]
    block_count: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    block_index: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), cdiv(N_CTX, BLOCK_SIZE_N)]
    sm_scale: float,
    block_size_M: int = 128,
    block_size_N: int = 64,
) -> torch.Tensor:
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.zeros_like(q)
    grid = (triton.cdiv(q.shape[2], block_size_M), q.shape[0] * q.shape[1], 1)
    dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16
    num_stages = 4 if block_size_M > 64 else 2
    triton_block_sparse_attn_kernel_v1[grid](
        q, k, v, sm_scale,
        block_count, block_index,
        o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        block_index.shape[-2], block_index.shape[-1],
        q.shape[0], q.shape[1], q.shape[2],
        BLOCK_M=block_size_M, BLOCK_N=block_size_N,
        BLOCK_DMODEL=Lk,
        dtype=dtype,
        num_warps=4, num_stages=num_stages,
    )

    return o


@triton.jit
def triton_dense_fwd_kernel(
    Q, K, V, sm_scale,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    dtype: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qo_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr, boundary_check=(1, 0), padding_option='zero')
    q = (q * qk_scale).to(dtype)
    # loop over k, v and update accumulator

    for start_n in range(0, N_CTX, BLOCK_N):
        n_mask = start_n + offs_n[None, :] < N_CTX
        # -- load k, v --
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option='zero')
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option='zero')
        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.where(n_mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        # -- compute scaling constant --
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # write back O
    acc /= l_i[:, None]
    O_block_ptr = tl.make_block_ptr(
        base=Out + qo_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_ok),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    tl.store(O_block_ptr, acc.to(dtype), boundary_check=(1, 0))


def triton_dense_forward(q, k, v, sm_scale, block_size_M=128, block_size_N=64) -> torch.Tensor:
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.zeros_like(q)
    grid = (triton.cdiv(q.shape[2], block_size_M), q.shape[0] * q.shape[1], 1)
    dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16
    num_stages = 4 if block_size_M > 64 else 3
    triton_dense_fwd_kernel[grid](
        q, k, v, sm_scale,
        o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1], q.shape[2],
        BLOCK_M=block_size_M, BLOCK_N=block_size_N,
        BLOCK_DMODEL=Lk,
        dtype=dtype,
        num_warps=4, num_stages=num_stages,
    )

    return o


def flash_attn_forward(q, k, v, sm_scale) -> torch.Tensor:
    return flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=sm_scale,
        causal=False,
    )


def torch_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sm_scale: float,
    mask: torch.Tensor,
) -> torch.Tensor:
    p = torch.einsum(f'bhmk, bhnk -> bhmn', query, key) * sm_scale
    if mask is not None:
        p = p.where(mask, -torch.inf)
    p_max = p.max(-1, keepdim=True).values
    p_max = torch.where(p_max < 0, 0.0, p_max)
    p_exp = torch.exp(p - p_max)
    s = p_exp / (p_exp.sum(-1, keepdim=True) + 1e-6)
    out = torch.einsum(f'bhmn, bhnk -> bhmk', s, value)
    return out


def make_block_index(block_mask: torch.Tensor):
    # block_mask = block_mask.contiguous()
    block_count = block_mask.sum(dim=-1, dtype=torch.int32)
    block_index = block_mask.to(torch.uint8).sort(dim=-1, descending=True).indices.to(torch.int32)
    # For triton_block_sparse_attn_kernel_v1, please remove the next 2 lines
    # prepend = torch.zeros((*block_index.shape[:-1], 1), dtype=block_index.dtype, device=block_index.device)
    # block_index = block_index.diff(dim=-1, prepend=prepend)
    return block_count, block_index


def make_full_mask(block_mask: torch.Tensor, context_size: int, block_size_M=64, block_size_N=64):
    batch_size, num_heads, num_rows, num_cols = block_mask.shape
    full_mask = block_mask.unsqueeze(-2).unsqueeze(-1).repeat((1, 1, 1, block_size_M, 1, block_size_N))
    full_mask = full_mask.reshape((batch_size, num_heads, num_rows * block_size_M, num_cols * block_size_N))
    return full_mask[:, :, :context_size, :context_size]


# def pad_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pad_M: int, pad_N: int):
#     pad_q = torch.nn.functional.pad(q, (0, 0, 0, pad_M, 0, 0, 0, 0))
#     pad_k = torch.nn.functional.pad(k, (0, 0, 0, pad_N, 0, 0, 0, 0))
#     pad_v = torch.nn.functional.pad(v, (0, 0, 0, pad_N, 0, 0, 0, 0))
#     return pad_q, pad_k, pad_v


def profile(fn, total_flops, tag, warmup=25, rep=100):
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if total_flops > 0:
        gflops = total_flops / ms * 1e-9
        print(f'  {tag} : {ms:.3f} ms | {gflops:.3f} GFLOP/s')
    else:
        print(f'  {tag} : {ms:.3f} ms')


def test_flash_attention(
    dtype=torch.float16,
    device="cuda",
    batch_size=4,
    num_heads=24,
    context_size=4096,
    head_dim=64,
    sparsity=0.5,
    block_size_M=128,
    block_size_N=64,
):
    print('==================================================')
    print(f'B={batch_size}, N={context_size}, H={num_heads}, D={head_dim}, SPARSITY={sparsity:.2f}')
    q = torch.randn((batch_size, num_heads, context_size, head_dim), dtype=dtype, device=device)
    k = torch.randn((batch_size, num_heads, context_size, head_dim), dtype=dtype, device=device)
    v = torch.randn((batch_size, num_heads, context_size, head_dim), dtype=dtype, device=device)
    sm_scale = head_dim ** -0.5
    dense_flops = 2. * batch_size * num_heads * context_size * context_size * head_dim

    num_rows = (context_size + block_size_M - 1) // block_size_M
    num_cols = (context_size + block_size_N - 1) // block_size_N

    block_mask = torch.rand((batch_size, num_heads, num_rows, num_cols), device=device) > sparsity
    block_count, block_index = make_block_index(block_mask)
    torch_build_index_fn = lambda: make_block_index(block_mask)
    profile(torch_build_index_fn, -1, ' torch-index ')

    ref_o_dense = torch_forward(q, k, v, sm_scale, mask=None)

    full_mask = make_full_mask(block_mask, context_size, block_size_M, block_size_N)
    ref_o_sparse = torch_forward(q, k, v, sm_scale, mask=full_mask)

    triton_dense_fn = lambda: triton_dense_forward(q, k, v, sm_scale, block_size_M, block_size_N)
    output = triton_dense_fn()
    torch.testing.assert_close(output, ref_o_dense, atol=1e-3, rtol=0)
    profile(triton_dense_fn, dense_flops, 'triton-dense ')

    triton_sparse_fn = lambda: triton_block_sparse_forward(q, k, v, block_count, block_index, sm_scale, block_size_M, block_size_N)
    output = triton_sparse_fn()
    torch.testing.assert_close(output, ref_o_sparse, atol=1e-3, rtol=0)
    profile(triton_sparse_fn, dense_flops * sparsity, 'triton-sparse')

    q = q.swapaxes(1, 2).contiguous()
    k = k.swapaxes(1, 2).contiguous()
    v = v.swapaxes(1, 2).contiguous()

    flash_fn = lambda: flash_attn_forward(q, k, v, sm_scale)
    output = flash_fn()
    output = output.swapaxes(1, 2).contiguous()
    torch.testing.assert_close(output, ref_o_dense, atol=1e-3, rtol=0)
    profile(flash_fn, dense_flops, ' flash-dense ')
    print('==================================================\n')


# torch.manual_seed(2024)

# test_flash_attention(num_heads=24, head_dim=64, context_size=4321, sparsity=0.5, block_size_M=128, block_size_N=64)
# test_flash_attention(num_heads=1, head_dim=64, context_size=32768, sparsity=0.5, block_size_M=128, block_size_N=64)


def block_sparse_flash_attention_forward(
    query: torch.Tensor,        # [BATCH, N_HEADS, N_CTX, D_HEAD]
    key: torch.Tensor,          # [BATCH, N_HEADS, N_CTX, D_HEAD]
    value: torch.Tensor,        # [BATCH, N_HEADS, N_CTX, D_HEAD]
    block_mask: torch.Tensor,   # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), cdiv(N_CTX, BLOCK_SIZE_N)], torch.bool
    block_size_M: int = 128,
    block_size_N: int = 64,
):
    sm_scale = query.shape[-1] ** -0.5
    block_count, block_index = make_block_index(block_mask)  # better move this line to build_mask()
    out = triton_block_sparse_forward(query, key, value, block_count, block_index, sm_scale, block_size_M, block_size_N)
    return out


if __name__ == '__main__':
    query = torch.randn((2, 24, 4250, 64), dtype=torch.float16, device='cuda')
    key = torch.randn((2, 24, 4250, 64), dtype=torch.float16, device='cuda')
    value = torch.randn((2, 24, 4250, 64), dtype=torch.float16, device='cuda')
    block_mask = torch.rand((2, 24, 34, 67), device="cuda") > 0.7

    out = block_sparse_flash_attention_forward(query, key, value, block_mask)

    prof = torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=2, warmup=10, active=10, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./logdir/kernel'),
    record_shapes=False,
    with_stack=False)
    prof.start()

    for i in range(22):
        prof.step()
        out = block_sparse_flash_attention_forward(query, key, value, block_mask)
    prof.stop()
