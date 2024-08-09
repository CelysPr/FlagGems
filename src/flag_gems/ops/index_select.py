import logging

import torch
import triton
import triton.language as tl

from ..utils import dim_compress, libentry


def cfggen():
    block_m = [1, 2, 4, 8]
    configs = [
        triton.Config({"BLOCK_M": m, "BLOCK_N": 1024}, num_warps=4) for m in block_m
    ]
    return configs


@libentry()
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def index_select_kernel(
    inp, out, M, N, index, index_len, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    rows_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    rows_mask = rows_offsets < M

    for off in range(0, index_len, BLOCK_N):
        cols_offsets = off + tl.arange(0, BLOCK_N)
        block_mask = rows_mask and (cols_offsets < N)
        out_mask = rows_mask and (cols_offsets < index_len)

        indices = tl.load(
            index + cols_offsets, mask=(cols_offsets < index_len), other=0
        )
        inp_off = rows_offsets * N + indices[None, :]
        out_off = rows_offsets * index_len + cols_offsets[None, :]

        selected = tl.load(inp + inp_off, mask=block_mask, other=0.0)
        tl.store(out + out_off, selected, mask=out_mask)


def index_select(inp, dim, index):
    logging.debug("GEMS INDEX SELECT")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.ndim <= 1, "Index should have dimension 1 or 0"
    assert ((i >= 0 and i < inp.size(dim)) for i in index), "Index out of range"

    if index.ndim == 0:
        index = index.unsqueeze(0)
    dim = dim % inp.ndim
    inp_shape = list(inp.shape)

    # with dim_compress
    inp = dim_compress(inp, dim)
    N = inp_shape[dim]
    M = inp.numel() // N
    index_len = index.numel()
    out_shape = list(inp.shape)
    out_shape[inp.ndim - 1] = index.numel()
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    index_select_kernel[grid](inp, out, M, N, index, index_len)
    if dim != out.ndim - 1:
        order = [i for i in range(out.ndim - 1)]
        order.insert(dim, out.ndim - 1)
        return out.permute(order).contiguous()
    else:
        return out
    """
    # with gather
    new_index_shape = [1] * inp.ndim
    new_index_shape[dim] = index.size(0)
    index_ = index.view(new_index_shape)
    new_index_shape = inp_shape
    new_index_shape[dim] = index.size(0)
    index_ = index_.expand(new_index_shape)
    return torch.gather(inp, dim, index_)
    """