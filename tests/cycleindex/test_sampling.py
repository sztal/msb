"""Tests copied from the
`original repo <https://github.com/milani/cycleindex>`_
"""
# pylint: disable=missing-function-docstring
import pytest
import numpy as np
from msb.cycleindex.sampling import nrsampling, set_sampler_seed


@pytest.mark.parametrize("G,size,expected,seed", [
    (
        np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]]),
        3,
        [0, 1, 2],
        123
    ),
    (
        np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
        3,
        [2],
        123
    ),
    (
        np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
        2,
        [1, 2],
        123
    ),
])
def test_nrsampling(G, size, expected, seed):
    set_sampler_seed(seed)
    assert (nrsampling(G, size) == expected).all()
