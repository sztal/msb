"""Sampling functions."""
import random
import numpy as np
import numba
from .utils import is_weakly_connected


__all__ = ["vxsampling", "nrsampling"]


@numba.njit
def set_sampler_seed(seed: int) -> None:
    """Set sampler random seed."""
    random.seed(seed)

@numba.njit
def _vxsampling(
    A: np.ndarray[tuple[int, int]],
    size: int,
    exact: bool = False
) -> tuple[list[int], np.ndarray[tuple[int]], np.ndarray[tuple[int]]]:
    subgraph = numba.typed.List.empty_list(numba.int64)
    allowed = np.full(len(A), True)
    neighbourhood = np.full(len(A), False)
    neighbourhood[random.randint(0, len(A) - 1)] = True

    # random vertex expansion
    while len(subgraph) < size:
        neighbours = np.where(neighbourhood & allowed)[0]
        if len(neighbours) == 0:
            if not exact:
                return subgraph, allowed, neighbourhood
            subgraph.clear()
            allowed = np.full(len(A), True)
            neighbourhood = np.full(len(A), False)
            neighbourhood[random.randint(0, len(A) - 1)] = True
        # np.random.choice is a bit slower.
        u_index = random.randint(0, len(neighbours) - 1)
        u = neighbours[u_index]
        allowed[u] = False
        neighbourhood[u] = True
        # direction is not important
        neighbourhood = neighbourhood + (A[u, :] != 0) + (A[:, u] != 0)
        subgraph.append(u)

    return subgraph, allowed, neighbourhood

@numba.njit
def vxsampling(
    A: np.ndarray[tuple[int, int]],
    size: int,
    exact: bool = False
) -> list[int]:
    """Vertex Expansion algorithm to sample connected induced subgraphs
    of a given ``size`` from a graph with the adjacency matrix ``A``.

    Parameters
    ----------
    G
        Adjacency matrix of the graph. It should not contain self-loops.
    size
        Size of subgraph to sample
    exact
        If True, the algorithm tries to find a subgraph that matches the size exactly.
        It means if the graph G does not include such subgraph, the function never returns.
        If False, the algorithm returns a subgraph that might be smaller than the required size,
        even if such subgraph exists in the graph.

    Returns
    -------
    vertices
        List of vertices that form the sampled subgraph.
    """
    subgraph, _, _ = _vxsampling(A, size, exact=exact)
    return np.array(list(sorted(subgraph)))

@numba.njit
def nrsampling(
    A: np.ndarray[tuple[int, int]],
    size: int,
    exact: bool = False
) -> list[int]:
    """Uses NRS algorithm (see notes) to uniformly sample connected induced
    subgraphs of a given ``size`` from a graph with the adjacency matrix ``A``.

    Parameters
    ----------
    G
        Adjacency matrix of the graph. It should not contain self-loops.
    size
        Size of subgraph to sample
    exact
        If ``True``, the algorithm tries to find a subgraph that matches the size exactly.
        It means if ``A`` does not include such subgraph, the function never returns.
        If ``False``, the algorithm returns a subgraph that might be smaller than the required size,
        even if such subgraph exists in the graph.

    Notes
    -----
    Lu X., Bressan S. (2012) Sampling Connected Induced Subgraphs Uniformly at Random.
    In: Ailamaki A., Bowers S. (eds) Scientific and Statistical Database Management.
    SSDBM 2012. Lecture Notes in Computer Science, vol 7338. Springer, Berlin, Heidelberg

    Returns
    -------
    vertices
        List of vertices that form the sampled subgraph.
    """
    # pylint: disable=too-many-locals
    subgraph, allowed, neighbourhood = _vxsampling(A, size, exact=exact)
    # Fix for bias toward subgraphs with higher clustering coef.
    i = int(size)
    neighbours = np.where(neighbourhood & allowed)[0]
    while len(neighbours) > 0:
        i += 1
        v_index = random.randint(0, len(neighbours) - 1)
        v = neighbours[v_index]
        alpha = random.random()
        if alpha < float(size) / i:
            u_index = random.randint(0, len(subgraph) - 1)

            s_prime = np.array(list(subgraph))
            s_prime[u_index] = v
            ki = s_prime.astype(np.int64)
            kj = s_prime.astype(np.int64)
            s_prime_adj = A[ki][:, kj]
            if is_weakly_connected(s_prime_adj):
                subgraph[u_index] = v

        allowed[v] = False
        neighbourhood = neighbourhood + (A[v, :] != 0) + (A[:, v] != 0)
        neighbours = np.where(neighbourhood & allowed)[0]

    return np.array(list(sorted(subgraph)))
