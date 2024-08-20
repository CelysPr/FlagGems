import logging

import torch
import triton
import triton.language as tl

from ..utils import pointwise_dynamic

try:
    from triton.language.extra.cuda.libdevice import pow
except ImportError:
    try:
        from triton.language.math import pow
    except ImportError:
        from triton.language.libdevice import pow

try:
    from triton.language.extra.cuda.libdevice import tanh as _tanh
except ImportError:
    try:
        from triton.language.math import tanh as _tanh
    except ImportError:
        from triton.language.libdevice import tanh as _tanh


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def tanh_forward(x):
    return _tanh(x.to(tl.float32))


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def tanh_backward(y, dy):
    return dy * (1.0 - y * y)

@triton.jit
def tanh_backward_custom_kernel(
    x_ptr: tl.tensor,  # *Pointer* to first input vector.
    y_ptr: tl.tensor,  # *Pointer* to second input vector.
    output_ptr: tl.tensor,  # *Pointer* to output vector.
    n_elements: int,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                        # NOTE: `constexpr` so it can be used as a shape value.
):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)

    #No need to add offset and mask, as its stride is 0
    y = tl.load(y_ptr)

    output = y * (1 - x * x)
    # Write output back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)

def tanh_backward_custom(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda

    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    tanh_backward_custom_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

class Tanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        logging.debug("GEMS TANH FORWARD")
        if A.requires_grad is True:
            out = tanh_forward(A.to(torch.float32))
            ctx.save_for_backward(out)
            return out.to(A.dtype)
        else:
            out = tanh_forward(A)
            return out

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS TANH BACKWARD")
        (out,) = ctx.saved_tensors
        
        is_grad_stride_0 = True
        for i in range(len(out_grad.stride())):
            if out_grad.stride()[i] != 0:
                is_grad_stride_0 = False
                break

        # temporay plan
        if (is_grad_stride_0) and (out_grad.numel() % 1024 == 0):
            in_grad = tanh_backward_custom(out, out_grad)
            return in_grad

        in_grad = tanh_backward(out, out_grad)
        return in_grad


def tanh(A):
    return Tanh.apply(A)
