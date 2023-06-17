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

def calc_ratio(
    diff: np.ndarray[tuple[int], np.floating],
    total: np.ndarray[tuple[int], np.floating]
) -> np.ndarray[tuple[int], np.floating]:
    """Calculate ratio of negative cycles to all cycles.

    Parameters
    ----------
    diff
        Array of differences between counts of positive and negative cycles
        of different lengths.
    total
        Array of the total counts of cycles of different lengths.
    """
    diff = diff.squeeze()
    total = total.squeeze()
    denom = -2*total
    denom[denom == 0] = 1
    return (diff - total) / denom
