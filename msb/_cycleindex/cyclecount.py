"""Cycle counting functions."""
import numpy as np
import numba
from .utils import is_symmetric


__all__ = ["cycle_count"]


@numba.njit
def prime_count(
    A: np.ndarray[tuple[int, int]],
    L0: int,
    subgraph: list[int],
    n_neighbours: list[int],
    primes: tuple[list[int], list[int]],
    directed: bool
) -> tuple[list[int], list[int]]:
    """Calculate the contribution to the combinatorial sieve of a
    given subgraph. This function is an implementation of the
    Eq. (2), extracting prime numbers from connected induced subgraphs

    Parameters
    ----------
    A
        Adjacency matrix of the graph, preferably sparse
    L0
        Maximum subgraph size
    subgraph
        Current subgraph, a list of vertices, further vertices are
        added to this list
    primes
        A tuple of two 1D complex arrays regrouping
        the contributions of all subgraphs considered earlier.
    directed
        If the graph is directed

    Returns
    -------
    tuple
        A tuple of two 1D complex arrays.
        Each show the contribution of all subgraphs so far
        and now including the contributuon of the subgraph passed to this function.
        The first is N_positive - N_negative, the next is N_positive + N_negative.
    """
    # pylint: disable=too-many-locals
    subgraph_size = len(subgraph)
    x = A[subgraph][:, subgraph].astype(np.complex128)
    x_p = np.abs(x).astype(np.complex128)
    if directed:
        xeig = np.linalg.eigvals(x)
        xeig_p = np.linalg.eigvals(x_p)
    else:
        xeig = np.linalg.eigvalsh(x).astype(np.complex128)
        xeig_p = np.linalg.eigvalsh(x_p).astype(np.complex128)
    xS = xeig**subgraph_size
    xS_p = xeig_p**subgraph_size
    mk = min(L0, n_neighbours + subgraph_size)

    binomial_coeff = 1
    for k in range(subgraph_size, mk):
        primes[0][k-1] += (-1)**k * binomial_coeff * (-1)**subgraph_size * sum(xS) / k
        primes[1][k-1] += (-1)**k * binomial_coeff * (-1)**subgraph_size * sum(xS_p) / k
        xS = xS * xeig
        xS_p = xS_p * xeig_p
        binomial_coeff *= (subgraph_size-k+n_neighbours) / (1-subgraph_size+k)
    primes[0][mk-1] += (-1)**mk * binomial_coeff * (-1)**subgraph_size * sum(xS) / mk
    primes[1][mk-1] += (-1)**mk * binomial_coeff * (-1)**subgraph_size * sum(xS_p) / mk
    return primes


@numba.njit
def recursive_subgraphs(
    A: np.ndarray[tuple[int, int]],
    Anw: np.ndarray[tuple[int, int]],
    L0: int,
    subgraph: list[int],
    allowed: np.ndarray[tuple[int], np.integer],
    primes: tuple[list[int], list[int]],
    neighbourhood: np.ndarray[tuple[int], np.integer],
    directed: bool
) -> tuple[list[int], list[int]]:
    """Find all the connected induced subgraphs of size up
    to ``L0`` of a graph known through its adjacency matrix
    ``A`` and containing the subgraph ``Subgraph``

    Parameters
    ----------
    A : numpy.ndarray
        Adjacency matrix
    Anw: numpy.ndarray
        Undirected unweighted equivalence of A
    L0: int
        Maximum subgraph size
    subgraph
        List of vertices that form current subgraph
    allowed
        1D indicator vector array of pruned vertices that may be
        considered for addition to the current subgraph to
        form a larger one.
    primes
        A tuple of two 1D complex arrays regrouping the contributions
        of all subgraphs considered earlier.
    neighbourhood
        Indicator vector of the vertices that are contained
        in the current subgraph or reachable via one edge
    directed
        Shows if "A" is directed or not

    Returns
    -------
    tuple
        Two 1D complex arrays regrouping the contributions
        of all the subgraphs found so far.
    """
    L = len(subgraph)
    n_neighbours = len(np.nonzero(neighbourhood)[0]) - L
    primes = prime_count(
        A, L0, np.array(list(subgraph)),
        n_neighbours, primes, directed
    )
    if L == L0:
        return primes

    neighbours = np.where(neighbourhood & allowed)[0]
    for j, _ in enumerate(neighbours):
        v = neighbours[j]
        if len(subgraph) > L:
            subgraph[L] = v
        else:
            subgraph.append(v)
        allowed[v] = False
        new_neighbourhood = neighbourhood + Anw[v, :]
        primes = recursive_subgraphs(
            A, Anw, L0, subgraph.copy(), allowed.copy(),
            primes, new_neighbourhood, directed
        )

    return primes


# @numba.njit
def cycle_count(
    A: np.ndarray[tuple[int, int]],
    L0: int
) -> tuple[list[int], list[int]]:
    """Count all simple cycles of length up to ``L0`` included on a
    graph whose adjacency matrix is ``A``.

    Parameters
    ----------
    A
        Adjacency matrix
    L0
        Length of cycles to count

    Returns
    -------
    tuple
        Two lists are returned. In each, index ``i`` is the number of primes
        of length ``i>=1`` up to ``L0``.
        The first list is the count of ``N_positive - N_negative``.
        The second one is the count of ``N_positive + N_negative``.
        Using these two lists, one can compute ``N_positive`` and ``N_negative``.
    """
    A = A.copy()
    primes = (
        np.full(L0, 0, dtype=np.complex128),
        np.full(L0, 0, dtype=np.complex128)
    )
    np.fill_diagonal(A, 0)

    if is_symmetric(A):
        Anw = A != 0
        directed = False
    else:
        Anw = (A != 0) | (A.T != 0)
        directed = True

    size = len(A)
    L0 = min(size, L0)

    allowed = np.full(size, True)
    for i in range(len(A)):
        allowed[i] = False
        neighbourhood = np.full(size, False)
        neighbourhood[i] = True
        neighbourhood += Anw[i, :]
        subgraph = numba.typed.List([i])
        primes = recursive_subgraphs(
            A, Anw, L0, subgraph, allowed.copy(),
            primes, neighbourhood, directed
        )

    return primes
