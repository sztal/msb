"""Tests copied from the
`original repo <https://github.com/milani/cycleindex>`_
"""
# pylint: disable=missing-function-docstring
import random
import numpy as np
from msb.cycleindex import balance_ratio

# Set random seed
random.seed(303)

gama_pos = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])

gama_neg = np.array([[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]])

gama = gama_pos - gama_neg


def test_balance_ratio_exact():
    ratio = balance_ratio(gama, 10, exact=True)
    expected = [0, 0, 0.1324, 0.2792, 0.3793, 0.4447, 0.4864, 0.511, 0.5233, 0.527]
    assert np.allclose(np.round(ratio, decimals=4), expected)

def test_balance_ratio_accuracy_serial():
    # We calculate with accuracy 0.01 but we test within 0.02 tolerance. That is because
    # the algorithm is approximate. We hope we reach the accuracy of 0.01 but it might not
    # be the case. At the same time, we expect it to be near the real values.
    expected = [0, 0, 0.1324, 0.2792, 0.3793]
    ratio = balance_ratio(gama, 5, exact=False, accuracy=0.01, parallel=False)
    assert np.allclose(np.round(ratio, decimals=4), expected, atol=0.05)

def test_balance_ratio_accuracy_parallel():
    expected = [0, 0, 0.1324, 0.2792, 0.3793]
    ratio = balance_ratio(gama, 5, exact=False, accuracy=0.01, parallel=True)
    assert np.allclose(np.round(ratio, decimals=4), expected, atol=0.05)

def test_balance_ratio_n_sample_serial():
    expected = [0, 0, 0.1324, 0.2792, 0.3793]
    ratio = balance_ratio(gama, 5, exact=False, n_samples=6000, parallel=False)
    assert np.allclose(np.round(ratio, decimals=4), expected, atol=0.05)

def test_balance_ratio_n_sample_parallel():
    expected = [0, 0, 0.1324, 0.2792, 0.3793]
    ratio = balance_ratio(gama, 5, exact=False, n_samples=6000, parallel=True)
    assert np.allclose(np.round(ratio, decimals=4), expected, atol=0.05)
