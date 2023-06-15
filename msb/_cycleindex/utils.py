"""Utility functions."""
from typing import Iterable
import numpy as np
from numba import njit


@njit
def clean_matrix(
    A: np.ndarray[tuple[int, int]]
) -> np.ndarray[tuple[int, int]]:
    """Removes all the vertices of graph which do not sustain any cycle
    by iteratively removing isolated vertices, sinks and sources until
    the matrix is invariant under such removal.

    Parameters
    ----------
    A
        Adjacency matrix

    Returns
    -------
    A
        The shape of this array is different from the input array
    """
    oldshape = (0, 0)
    while oldshape != A.shape:
        oldshape = A.shape
        x = A.sum(axis=0) == 0
        A = A[~x][:, ~x]
        x = A.sum(axis=1) == 0
        A = A[~x][:, ~x]
    return A

@njit
def is_symmetric(A: np.ndarray[tuple[int, int]]) -> bool:
    """Checks if matrix ``A`` is symmetric

    Parameters
    ----------
    A
        Adjacency matrix
    **kwds
        Passed to :func:`numpy.allclose`.
    """
    return (A.ndim == 2) and (A.shape[0] == A.shape[1]) \
        and np.all(np.abs(A - A.T) <= 10e-9)

@njit
def dfs(
    A: np.ndarray[tuple[int, int]],
    root: int = 0
) -> Iterable[int]:
    """Depth-first search.

    Parameters
    ----------
    A
        Adjacency matrix of a graph.
    root
        Id of the root (starting) node.
    seen
        Sequence of already visited nodes.

    Yields
    ------
    i
        Vertex index.
    """
    stack = [root]
    seen = set()
    while stack:
        current = stack.pop()
        if current not in seen:
            if seen or current != root:
                seen.add(current)
                yield current
            for i in np.nonzero(A[current])[0][::-1]:
                if i not in seen:
                    stack.append(i)

@njit
def is_weakly_connected(A: np.ndarray[tuple[int, int]]) -> bool:
    """Check if graph is weakly-connected.

    Parameters
    ----------
    A
        Adjacency matrix of a graph.
    """
    # Undirected structural version of G
    _A = ((A != 0) | (A.T != 0)).astype(np.int64)
    vcount = 0
    for _ in dfs(_A):
        vcount += 1
    return vcount == len(_A)

def ndim(array):
    """Stupid."""
    count = 0
    if isinstance(array, list):
        count = 1 + ndim(array[0])
    return count

def calc_ratio(plus_minus, plus_plus):
    """Calculate ratio of something (?)."""
    if ndim(plus_minus) == 1:
        plus_minus = np.expand_dims(plus_minus, 0)
        plus_plus = np.expand_dims(plus_plus, 0)
    elif ndim(plus_minus) == 3:
        plus_minus = np.concatenate(plus_minus)
        plus_plus = np.concatenate(plus_plus)

    plus_minus = np.mean(plus_minus, axis=0)
    plus_plus = np.mean(plus_plus, axis=0)
    pos = (plus_plus + plus_minus) / 2
    neg = plus_plus - pos
    # avoid nan and inf
    plus_plus[np.isclose(plus_plus, 0)] = 1
    ratio = neg / plus_plus
    return ratio
#     plus_minus = np.array(plus_minus)
#     plus_plus = np.array(plus_plus)
#     if plus_minus.ndim == 1:
#         plus_minus = plus_minus[None, ...]
#         plus_plus = plus_plus[None, ...]
#     elif plus_minus.ndim == 3:
#         plus_minus = plus_minus[0]
#         plus_plus = plus_plus[0]
#     pm = plus_minus.astype(np.float64)
#     pp = plus_plus.astype(np.float64)

#     pm = pm.mean(axis=0)
#     pp = pp.mean(axis=0)
#     pos = (pp + pm) / 2
#     neg = pp - pos
#     # avoid nan and inf
#     pp[np.abs(pp) <= 10e-9] = 1
#     ratio = neg / pp
#     return ratio
