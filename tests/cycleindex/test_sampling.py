"""Tests copied from the
`original repo <https://github.com/milani/cycleindex>`_
"""
# pylint: disable=missing-function-docstring
import random
import pytest
import numpy as np
from msb.cycleindex.sampling import nrsampling


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
        [0, 1],
        123
    ),
    (
        np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
        2,
        [2, 1],
        123
    ),
])
def test_nrsampling(G, size, expected, seed):
    random.seed(seed)
    assert nrsampling(G, size) == expected
