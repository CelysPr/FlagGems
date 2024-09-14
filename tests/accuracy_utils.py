import itertools

import torch

from .conftest import TO_CPU


def SkipTorchVersion(skip_pattern):
    cmp = skip_pattern[0]
    assert cmp in ("=", "<", ">")
    try:
        M, N = skip_pattern[1:].split(".")
        M, N = int(M), int(N)
    except Exception:
        raise "Cannot parse version number."
    major, minor = torch.__version__.split(".")[:2]
    major, minor = int(major), int(minor)

    if cmp == "=":
        return major == M and minor == N
    elif cmp == "<":
        return (major, minor) < (M, N)
    else:
        return (major, minor) > (M, N)


INT16_MIN = torch.iinfo(torch.int16).min
INT16_MAX = torch.iinfo(torch.int16).max
INT32_MIN = torch.iinfo(torch.int32).min
INT32_MAX = torch.iinfo(torch.int32).max

RESOLUTION = {
    torch.bool: 0,
    torch.int16: 0,
    torch.int32: 0,
    torch.float16: 1e-3,
    torch.float32: 1.3e-6,
    torch.bfloat16: 0.016,
}

sizes_one = [1]
sizes_pow_2 = [2**d for d in range(4, 11, 2)]
sizes_noalign = [d + 17 for d in sizes_pow_2]
sizes_1d = sizes_one + sizes_pow_2 + sizes_noalign
sizes_2d_nc = [1] if TO_CPU else [1, 16, 64, 1000]
sizes_2d_nr = [1] if TO_CPU else [1, 5, 1024]

UT_SHAPES_1D = list((n,) for n in sizes_1d)
UT_SHAPES_2D = list(itertools.product(sizes_2d_nr, sizes_2d_nc))
POINTWISE_SHAPES = (
    [(2, 19, 7)]
    if TO_CPU
    else [(1,), (1024, 1024), (20, 320, 15), (16, 128, 64, 60), (16, 7, 57, 32, 29)]
)
SPECIAL_SHAPES = (
    [(2, 19, 7)]
    if TO_CPU
    else [(1,), (1024, 1024), (20, 320, 15), (16, 128, 64, 1280), (16, 7, 57, 32, 29)]
)
DISTRIBUTION_SHAPES = [(20, 320, 15)]
REDUCTION_SHAPES = [(2, 32)] if TO_CPU else [(1, 2), (4096, 256), (200, 40999, 3)]
REDUCTION_SMALL_SHAPES = [(1, 32)] if TO_CPU else [(1, 2), (4096, 256), (200, 2560, 3)]
STACK_SHAPES = [
    [(16,), (16,)],
    [(16, 256), (16, 256)],
    [(20, 320, 15), (20, 320, 15), (20, 320, 15)],
]

FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
ALL_FLOAT_DTYPES = FLOAT_DTYPES + [torch.float64]
INT_DTYPES = [torch.int16, torch.int32]
ALL_INT_DTYPES = INT_DTYPES + [torch.int64]
BOOL_TYPES = [torch.bool]

SCALARS = [0.001, -0.999, 100.001, -111.999]
STACK_DIM_LIST = [-2, -1, 0, 1]


def to_reference(inp, upcast=False):
    if inp is None:
        return None
    ref_inp = inp
    if TO_CPU:
        ref_inp = ref_inp.to("cpu")
    if upcast:
        ref_inp = ref_inp.to(torch.float64)
    return ref_inp


def gems_assert_close(a, b, dtype, equal_nan=False, reduce_dim=1):
    if TO_CPU:
        a = a.to("cpu")
    b = b.to(dtype)
    atol = 1e-4 * reduce_dim
    rtol = RESOLUTION[dtype]
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol, equal_nan=equal_nan)


def gems_assert_close_groupnorm(a, b, dtype, equal_nan=False, reduce_dim=1):
    if TO_CPU:
        a = a.to("cpu")
    b = b.to(dtype)
    atol = 1e-4 * reduce_dim
    rtol = RESOLUTION[dtype]
    if dtype == torch.float16:
        atol = 1e-2
    if dtype == torch.bfloat16:
        atol = 2e-1
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol, equal_nan=equal_nan)


def gems_assert_close_layernorm(a, b, dtype, equal_nan=False, reduce_dim=1):
    if TO_CPU:
        a = a.to("cpu")
    b = b.to(dtype)
    atol = 1e-4 * reduce_dim
    rtol = RESOLUTION[dtype]
    if dtype == torch.float16:
        atol = 1e-2
    if dtype == torch.bfloat16:
        atol = 5e-2
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol, equal_nan=equal_nan)


def gems_assert_equal(a, b):
    if TO_CPU:
        a = a.to("cpu")
    assert torch.equal(a, b)


def unsqueeze_tuple(t, max_len):
    for _ in range(len(t), max_len):
        t = t + (1,)
    return t


def unsqueeze_tensor(inp, max_ndim):
    for _ in range(inp.ndim, max_ndim):
        inp = inp.unsqueeze(-1)
    return inp


# XPU Setting
# some reduction op needs specific BLOCK_SIZE params
# REDUCTION_SHAPES = [(4096, 256)]
XPU_REDUCTION_SHAPES_M = [(12, 256)]  # SHAPE[0] <= CLUSTE_NUM   GRIDX <= CLUSTE_NUM
XPU_REDUCTION_SHAPES_N = [(4096, 1)]  # SHAPE[1] == 1            GRIDY == 1
XPU_POINTWISE_2D_SHAPES_8192 = [
    (16, 1024, 8)
]  # SHAPE[-1] * SHAPE[-2] <= 8192(core_num * buffer_size limit)

ALL_FLOAT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
ALL_INT_DTYPES = [torch.int32, torch.int16]  # miss torch.int64

# vendor-test shape
KEY_OPS_SHAPES = [(1024, 32), (1024, 96), (1024, 8192), (1024, 20480), (1024, 32768)]
