import logging

import triton
import triton.language as tl

from ..utils import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def cos_func(x):
    return tl.cos(x.to(tl.float32))


def cos(A):
    logging.debug("GEMS COS")
    return cos_func(A)
