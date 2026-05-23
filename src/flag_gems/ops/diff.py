import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

Tensor = torch.Tensor


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("diff_1d"),
    key=["N"],
)
@triton.jit
def diff_kernel_1d(
    in_ptr,
    out_ptr,
    N,
    N_bound,
    BLOCK_DIFF: tl.constexpr,
):
    pid = tle.program_id(0)

    in_offsets = pid * BLOCK_DIFF + tl.arange(0, BLOCK_DIFF)
    mask_in = in_offsets < N_bound - 1
    in_block = tl.load(in_ptr + in_offsets, mask_in)
    next_block = tl.load(in_ptr + in_offsets + 1, mask_in)
    tl.store(out_ptr + in_offsets, next_block - in_block, mask_in)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("diff"),
    key=["M", "N"],
)
@triton.jit
def diff_kernel_2d(
    in_ptr,
    out_ptr,
    M,
    N,
    N_bound,
    BLOCK_M: tl.constexpr,
    BLOCK_DIFF: tl.constexpr,
):
    pid_M = tle.program_id(0)
    pid_diff = tle.program_id(1)

    M_offsets = pid_M * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_M = M_offsets < M

    in_offsets_diff = pid_diff * BLOCK_DIFF + tl.arange(0, BLOCK_DIFF)
    mask_in_diff = in_offsets_diff < N_bound - 1

    in_offsets = M_offsets[:, None] * N + in_offsets_diff[None, :]
    mask_in = mask_M[:, None] & mask_in_diff[None, :]

    in_block = tl.load(in_ptr + in_offsets, mask_in)
    next_block = tl.load(in_ptr + in_offsets + 1, mask_in)
    tl.store(out_ptr + in_offsets, next_block - in_block, mask_in)


def diff(
    inp: Tensor,
    n: int = 1,
    dim: int = -1,
    prepend: Optional[Tensor] = None,
    append: Optional[Tensor] = None,
) -> Tensor:
    """Compute the n-th forward difference along the given dimension.

    out[i] = input[i + 1] - input[i], applied recursively n times along `dim`.

    Matches ``torch.diff`` semantics:

    * ``n < 0`` raises ``RuntimeError`` ("order must be non-negative ...").
    * 0-d input raises ``RuntimeError``.
    * ``n == 0`` returns ``inp.clone()`` *before* applying prepend/append
      (a fresh tensor, not aliased to the input).
    * ``n >= shape[dim]`` (after prepend/append) returns an empty tensor
      with that dim set to 0.
    * Otherwise computes the n-th forward difference along ``dim``.
    """
    logger.debug("GEMS DIFF")

    if n < 0:
        raise RuntimeError(f"order must be non-negative but got {n}")

    if inp.ndim == 0:
        raise RuntimeError("diff expects input to be at least one-dimensional")

    if n == 0:
        # torch.diff short-circuits at n == 0 *before* applying prepend/append,
        # returning a fresh clone of the original input.
        return inp.clone()

    dim = dim % inp.ndim

    # Apply prepend / append by concatenation along `dim`.
    if prepend is not None or append is not None:
        tensors_to_cat = []
        if prepend is not None:
            tensors_to_cat.append(prepend)
        tensors_to_cat.append(inp)
        if append is not None:
            tensors_to_cat.append(append)
        inp = torch.cat(tensors_to_cat, dim=dim)

    reduce_len = inp.shape[dim]
    if n >= reduce_len:
        out_shape = list(inp.shape)
        out_shape[dim] = 0
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    # Move target dim to the innermost position and make contiguous, so that
    # both kernels operate on a stride-1 axis.
    inp = dim_compress(inp, dim)
    N = reduce_len
    M = inp.numel() // N

    # A single full-width buffer is reused via copy_ across the n iterations.
    # Each iteration writes only the first (cur_in_diff_len - 1) columns; the
    # remaining tail is never read by subsequent iterations because of the
    # shrinking load mask, so torch.empty is safe.
    output = torch.empty_like(inp)

    with torch_device_fn.device(inp.device):
        for step in range(n):
            cur_in_diff_len = N - step
            if inp.ndim == 1:
                grid = lambda meta: (  # noqa: E731
                    triton.cdiv(cur_in_diff_len, meta["BLOCK_DIFF"]),
                )
                diff_kernel_1d[grid](inp, output, N, cur_in_diff_len)
            else:
                grid = lambda meta: (  # noqa: E731
                    triton.cdiv(M, meta["BLOCK_M"]),
                    triton.cdiv(cur_in_diff_len, meta["BLOCK_DIFF"]),
                )
                diff_kernel_2d[grid](inp, output, M, N, cur_in_diff_len)
            inp.copy_(output)

    output = output[..., : N - n].contiguous()
    output = torch.moveaxis(output, -1, dim)
    return output
