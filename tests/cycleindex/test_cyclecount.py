"""Tests copied from the
`original repo <https://github.com/milani/cycleindex>`_
"""
# pylint: disable=missing-function-docstring
import pytest
import numpy as np
from msb.cycleindex.cyclecount import cycle_count


@pytest.mark.parametrize("A, expected", [
    (
        np.array([[0, 0.5, 0],
                  [0, 0, -0.5],
                  [0.5, 0, 0]]),
        ([0, 0, -0.12500, 0, 0, 0, 0], [0, 0, 0.12500, 0, 0, 0, 0])
    ),
    (
        np.array([[0, 0.5, 0.5, 0],
                  [0.5, 0, 0.5, 0],
                  [0.5, 0.5, 0, 0.4],
                  [0, 0, 0.4, 0]]),
        ([0, 0.91, 0.25, 0, 0, 0, 0], [0, 0.91, 0.25, 0, 0, 0, 0])
    ),
    (
        np.array([[0, 0.5, 0, 0, 0, 0, 0],
                  [0.5, 0, 0.4, 0.4, 0, 0, 0],
                  [0, 0.4, 0, -0.5, 0.1, 0, 0],
                  [0, 0.4, -0.5, 0, 0, 0.6, 0],
                  [0, 0, 0.1, 0, 0, 0, 0.8],
                  [0, 0, 0, 0.6, 0, 0, 0.7],
                  [0, 0, 0, 0, 0.8, 0.7, 0]]),
        ([0, 2.3200, -0.1600, 0, -0.0336, 0.010752, 0], [0, 2.32, 0.16, 0, 0.0336, 0.010752, 0])
    ),
    (
        np.array([[0, 1, 0, 0, 0, 0, 0],
                  [1, 0, 1, 1, 0, 0, 0],
                  [0, 1, 0, 1, 1, 0, 0],
                  [0, 1, -1, 0, 0, 1, 0],
                  [0, 0, 1, 0, 0, 0, 1],
                  [0, 0, 0, 1, 0, 0, 1],
                  [0, 0, 0, 0, 1, 1, 0]]),
        ([0, 6, 0, 0, 0, 2, 0], [0, 8, 2, 0, 2, 2, 0])
    )
])
def test_cycle_count(A, expected):
    assert np.allclose(cycle_count(A, 7), expected)
