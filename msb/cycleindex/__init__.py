"""Reimplementation of the `Cycle Index <https://github.com/milani/cycleindex>`_
package using Numba LLVM JIT to make the implementation significantly more performant.
"""
from typing import Optional, Any
import numpy as np
from .cyclecount import cycle_count, cycle_count_sample
from .utils import calc_ratio
from .sampling import nrsampling, vxsampling


def balance_ratio(
    A: np.ndarray[tuple[int, int]],
    L0: int,
    *,
    sample_size: Optional[int] = None,
    **kwds: Any
) -> np.ndarray[tuple[int], float]:
    """Calculate balance ratios (negative cycles / all cycles)
    for cycles lengths up to ``L0``.

    Parameters
    ----------
    A
        Adjacency matrix
    L0
        Length of cycles to count
    sample_size
        Number of samples to draw.
        Calculate exact cycle counts when falsy or negative.
    **kwds
        Passed to :func:`msb.cycleindex.cyclecount.cycle_count_sample`
        when using sampling.
    """
    if not sample_size or sample_size < 0:
        diff, total = cycle_count(A, L0)
    else:
        diff, total = cycle_count_sample(A, L0, sample_size, **kwds)
    return calc_ratio(diff, total)
