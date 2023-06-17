import numpy as np
from .utils import is_symmetric


def prime_count(A, L0, Subgraph, NeighboursNumber, Primes, Directed):
    """
    Calculates the contribution to the combinatorial sieve of a
    given subgraph. This function is an implementation of the
    Eq. (2), extracting prime numbers from connected induced subgraphs

    Parameters
    ----------
    A : numpy.ndarray
        Adjacency matrix of the graph, preferably sparse
    L0: int
        Maximum subgraph size
    Subgraph: list
        Current subgraph, a list of vertices, further vertices are
        added to this list
    Primes: tuple
        A tuple of two lists regrouping the contributions of all subgraphs
        considered earlier
    Directed: bool
        If the graph is directed

    Returns
    -------
    tuple
        A tuple of two lists. Each show the contribution of all subgraphs so far
        and now including the contributuon of the subgraph passed to this function.
        The first is N_positive - N_negative, the next is N_positive + N_negative.
    """
    eigvals = np.linalg.eigvals if Directed else np.linalg.eigvalsh

    SubgraphSize = len(Subgraph)
    x = A[np.ix_(Subgraph, Subgraph)]
    x_p = np.abs(x)
    xeig = eigvals(x)
    xeig_p = eigvals(x_p)
    xS = np.power(xeig, SubgraphSize)
    xS_p = np.power(xeig_p, SubgraphSize)
    mk = min(L0, NeighboursNumber + SubgraphSize)

    BinomialCoeff = 1
    for k in range(SubgraphSize, mk):
        Primes[0][k - 1] += (-1) ** k * BinomialCoeff * (-1) ** SubgraphSize * sum(xS) / k
        Primes[1][k - 1] += (-1) ** k * BinomialCoeff * (-1) ** SubgraphSize * sum(xS_p) / k
        xS = xS * xeig
        xS_p = xS_p * xeig_p
        BinomialCoeff = BinomialCoeff * (SubgraphSize - k + NeighboursNumber) / (1 - SubgraphSize + k)
    Primes[0][mk - 1] += (-1)**mk * BinomialCoeff * (-1)**SubgraphSize * sum(xS) / mk
    Primes[1][mk - 1] += (-1)**mk * BinomialCoeff * (-1)**SubgraphSize * sum(xS_p) / mk
    return Primes


def recursive_subgraphs(A, Anw, L0, Subgraph, AllowedVert, Primes, Neighbourhood, Directed):
    """
    Finds all the connected induced subgraphs of size up
    to "L0" of a graph known through its adjacency matrix
    "A" and containing the subgraph "Subgraph"

    Parameters
    ----------
    A : numpy.ndarray
        Adjacency matrix
    Anw: numpy.ndarray
        Undirected unweighted equivalence of A
    L0: int
        Maximum subgraph size
    Subgraph: list
        List of vertices that form current subgraph
    AllowedVert: list
        Indicator vector of pruned vertices that may be
        considered for addition to the current subgraph to
        form a larger one
    Neighbourhood: list
        Indicator vector of the vertices that are contained
        in the current subgraph or reachable via one edge
    Directed: bool
        Shows if "A" is directed or not

    Returns
    -------
    tuple
        Two lists regrouping the contribution of all the subgraphs found so far
    """
    L = len(Subgraph)
    NeighboursNumber = len(np.nonzero(Neighbourhood)[0]) - L
    Primes = prime_count(A, L0, Subgraph, NeighboursNumber, Primes, Directed)
    if L == L0:
        return Primes

    Neighbours = np.where(np.array(Neighbourhood) & np.array(AllowedVert))[0]
    for j in range(len(Neighbours)):
        v = Neighbours[j]
        if len(Subgraph) > L:
            Subgraph[L] = v
        else:
            Subgraph.append(v)
        AllowedVert[v] = False
        newNeighbourhood = Neighbourhood + Anw[v, :]
        Primes = recursive_subgraphs(A, Anw, L0, Subgraph[:], AllowedVert[:], Primes, newNeighbourhood[:], Directed)

    return Primes


def cycle_count(A, L0):
    """
    Counts all simple cycles of length up to L0 included on a
    graph whose adjacency matrix is A.

    Parameters
    ----------
    A : numpy.ndarray
        Adjacency matrix
    L0: int
        Length of cycles to count

    Returns
    -------
    tuple
        Two lists are returned. In each, index "i" is the number of primes of length i>=1 up to L0
        The first list is the count of N_positive - N_negative. The second one is the count of
        N_positive + N_negative. Using these two lists, one can compute N_positive and N_negative.
    """
    Primes = ([0] * L0, [0] * L0)
    np.fill_diagonal(A, 0)

    if is_symmetric(A):
        Anw = A != 0
        directed = False
    else:
        Anw = (A != 0) | (A.transpose(1, 0) != 0)
        directed = True

    Size = len(A)
    if L0 > Size:
        L0 = Size

    AllowedVert = [True] * Size
    for i in range(len(A)):
        AllowedVert[i] = False
        Neighbourhood = [False] * Size
        Neighbourhood[i] = True
        Neighbourhood = Neighbourhood + Anw[i, :]
        Primes = recursive_subgraphs(A, Anw, L0, [i], AllowedVert[:], Primes, Neighbourhood[:], directed)

    return Primes

__all__ = ['cycle_count']
