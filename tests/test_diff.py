import pytest
import torch

import flag_gems

from .accuracy_utils import (
    FLOAT_DTYPES,
    INT_DTYPES,
    gems_assert_close,
    gems_assert_equal,
    to_reference,
)
from .conftest import QUICK_MODE

DIFF_SHAPES = [(1024,), (100, 200), (10, 20, 30), (16, 128, 64, 60)]
DIFF_DIMS = [0, -1]
DIFF_N_VALUES = [1, 2, 3]

if QUICK_MODE:
    DIFF_SHAPES = DIFF_SHAPES[:2]
    DIFF_N_VALUES = [1, 2]


def _randn(shape, dtype, device):
    if dtype in INT_DTYPES:
        return torch.randint(low=-10, high=11, size=shape, dtype=dtype, device=device)
    return torch.randn(shape, dtype=dtype, device=device)


@pytest.mark.diff
@pytest.mark.parametrize("shape", DIFF_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
def test_diff(shape, dtype):
    inp = _randn(shape, dtype, flag_gems.device)
    ref_inp = to_reference(inp)
    ref_out = torch.diff(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.diff(inp)
    if dtype in INT_DTYPES:
        gems_assert_equal(res_out, ref_out)
    else:
        gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.diff
@pytest.mark.parametrize("shape", DIFF_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
@pytest.mark.parametrize("dim", DIFF_DIMS)
def test_diff_dim(shape, dtype, dim):
    if len(shape) == 1 and dim != 0 and dim != -1:
        pytest.skip("dim out of range for 1d input")
    inp = _randn(shape, dtype, flag_gems.device)
    ref_inp = to_reference(inp)
    ref_out = torch.diff(ref_inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.diff(inp, dim=dim)
    if dtype in INT_DTYPES:
        gems_assert_equal(res_out, ref_out)
    else:
        gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.diff
@pytest.mark.parametrize("shape", [(1024,), (100, 200)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
@pytest.mark.parametrize("n", DIFF_N_VALUES)
def test_diff_n(shape, dtype, n):
    inp = _randn(shape, dtype, flag_gems.device)
    ref_inp = to_reference(inp)
    ref_out = torch.diff(ref_inp, n=n)
    with flag_gems.use_gems():
        res_out = torch.diff(inp, n=n)
    if dtype in INT_DTYPES:
        gems_assert_equal(res_out, ref_out)
    else:
        gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.diff
@pytest.mark.parametrize("shape", [(100,), (50, 60)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_diff_prepend_append(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    prepend_shape = shape[:-1] + (3,) if len(shape) > 1 else (3,)
    append_shape = shape[:-1] + (2,) if len(shape) > 1 else (2,)
    prepend = torch.randn(prepend_shape, dtype=dtype, device=flag_gems.device)
    append = torch.randn(append_shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)
    ref_prepend = to_reference(prepend)
    ref_append = to_reference(append)

    ref_out = torch.diff(ref_inp, prepend=ref_prepend)
    with flag_gems.use_gems():
        res_out = torch.diff(inp, prepend=prepend)
    gems_assert_close(res_out, ref_out, dtype)

    ref_out = torch.diff(ref_inp, append=ref_append)
    with flag_gems.use_gems():
        res_out = torch.diff(inp, append=append)
    gems_assert_close(res_out, ref_out, dtype)

    ref_out = torch.diff(ref_inp, prepend=ref_prepend, append=ref_append)
    with flag_gems.use_gems():
        res_out = torch.diff(inp, prepend=prepend, append=append)
    gems_assert_close(res_out, ref_out, dtype)


# --- Edge cases ----------------------------------------------------------


@pytest.mark.diff
def test_diff_n_zero_returns_clone():
    """n=0 must return a fresh tensor (not aliased to input), matching torch."""
    inp = torch.randn(8, device=flag_gems.device)
    with flag_gems.use_gems():
        out = torch.diff(inp, n=0)
    assert out.data_ptr() != inp.data_ptr()
    torch.testing.assert_close(out, inp)


@pytest.mark.diff
def test_diff_n_zero_ignores_prepend_append():
    """n=0 short-circuits before prepend/append, matching torch behavior."""
    inp = torch.randn(4, device=flag_gems.device)
    pre = torch.randn(2, device=flag_gems.device)
    app = torch.randn(3, device=flag_gems.device)
    ref = torch.diff(
        to_reference(inp), n=0, prepend=to_reference(pre), append=to_reference(app)
    )
    with flag_gems.use_gems():
        out = torch.diff(inp, n=0, prepend=pre, append=app)
    torch.testing.assert_close(out, ref)
    # torch returns just the input (shape unchanged), ignoring pre/app at n=0.
    assert out.shape == inp.shape


@pytest.mark.diff
def test_diff_negative_n_raises():
    inp = torch.randn(8, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        with flag_gems.use_gems():
            torch.diff(inp, n=-1)


@pytest.mark.diff
def test_diff_zero_dim_raises():
    inp = torch.tensor(5.0, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        with flag_gems.use_gems():
            torch.diff(inp)


@pytest.mark.diff
def test_diff_n_ge_size_returns_empty():
    """When n >= size along dim, torch returns an empty tensor of matching shape."""
    inp = torch.randn(5, device=flag_gems.device)
    with flag_gems.use_gems():
        out = torch.diff(inp, n=5)
    assert out.numel() == 0
    assert out.shape == (0,)
