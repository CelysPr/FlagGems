import logging

import triton
import triton.language as tl

from ..utils import pointwise_dynamic


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def gelu_none(x):
    scale: tl.constexpr = 0.7071067811
    output = 0.5 * x * (1 + tl.extra.mlu.libdevice.erf(x * scale))
    return output


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def gelu_tanh(x):
    output = (
        0.5
        * x
        * (
            1
            + tl.extra.mlu.libdevice.tanh(
                x * 0.79788456 * (1 + 0.044715 * tl.extra.mlu.libdevice.pow(x.to(tl.float32), 2))
            )
        )
    )
    return output


def gelu(A, *, approximate="none"):
    logging.debug("GEMS GELU")
    if approximate == "tanh":
        return gelu_tanh(A)
    else:
        return gelu_none(A)
