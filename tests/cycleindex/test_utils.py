"""Tests copied from the
`original repo <https://github.com/milani/cycleindex>`_
"""
# pylint: disable=missing-function-docstring,line-too-long
import pytest
import numpy as np
from msb.cycleindex.utils import clean_matrix, is_symmetric
from msb.cycleindex.utils import is_weakly_connected, calc_ratio


@pytest.mark.parametrize("A, out", [
    (
        np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0]]),
        np.array([[1]])
    ),
    (
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    ),
    (
        np.array([[0, 0.5, 0, 0], [0, 0, 0.5, 0.4], [0, -0.1, 0, 0.2], [0, 0.2, 0.2, 0]]),
        np.array([[0, 0.5, 0.4], [-0.1, 0, 0.2], [0.2, 0.2, 0]])
    )
])
def test_clean_matrix(A, out):
    assert not np.any(clean_matrix(A) != out)


@pytest.mark.parametrize("A, expected", [
    (
        np.array([[0, 0.5, 0], [0, 0, 0.5], [0.4, 0, 0]]),
        False
    ),
    (
        np.array([[0, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0]]),
        True
    )
])
def test_is_symmetric(A, expected):
    assert is_symmetric(A) == expected


@pytest.mark.parametrize("A, expected", [
    (
        np.array([[0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]]),
        True
    ),
    (
        np.array([[0, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0]]),
        False
    )
])
def test_is_weakly_connected(A, expected):
    assert is_weakly_connected(A) == expected


@pytest.mark.parametrize("counts, expected", [
    (
        ([[1, 1, 1]], [[2, 2, 2]]),
        [0.25, 0.25, 0.25]
    ),
    (
        ([0.5, 0.5, 0.5, 0.5], [2, 2, 2, 2]),
        [0.375, 0.375, 0.375, 0.375]
    ),
    (
        ([[[1, 1, 1], [1, 1, 1]]], [[[2, 2, 2], [2, 2, 2]]]),
        [0.25, 0.25, 0.25]
    )
])
def test_calc_ratio(counts, expected):
    assert np.allclose(calc_ratio(*counts), expected)
