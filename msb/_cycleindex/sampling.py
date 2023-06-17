import numpy as np
import random
from .utils import is_weakly_connected


def _vxsampling(G, size, exact=False):
    subgraph = []
    allowed = np.array([True] * len(G))
    neighbourhood = np.array([False] * len(G))

    neighbourhood[random.randint(0, len(G) - 1)] = True

    # random vertex expansion
    while len(subgraph) < size:
        neighbours = np.where(neighbourhood & allowed)[0]
        if len(neighbours) == 0:
            if not exact:
                return subgraph, allowed, neighbourhood
            else:
                subgraph = []
                allowed = np.array([True] * len(G))
                neighbourhood = np.array([False] * len(G))
                neighbourhood[random.randint(0, len(G) - 1)] = True
                continue

        # np.random.choice is a bit slower.
        u_index = random.randint(0, len(neighbours) - 1)
        u = neighbours[u_index]

        allowed[u] = False
        neighbourhood[u] = True
        neighbourhood = neighbourhood + (G[u, :] != 0) + (G[:, u] != 0)  # direction is not important
        subgraph.append(u)

    return subgraph, allowed, neighbourhood


def vxsampling(G, size, exact=False):
    """
    Uses Vertex Expansion algorithm to sample connected induced subgraphs of size "size" from graph "G".

    Parameters
    ----------
    G : numpy.ndarray
        Adjacency matrix of the graph. It should not contain self-loops.
    size : int
        Size of subgraph to sample
    exact : bool
        If True, the algorithm tries to find a subgraph that matches the size exactly.
        It means if the graph G does not include such subgraph, the function never returns.
        If False, the algorithm returns a subgraph that might be smaller than the required size,
        even if such subgraph exists in the graph.

    Returns
    -------
    list
        List of vertices that form the sampled subgraph.
    """
    subgraph, _, _ = _vxsampling(G, size, exact)
    return subgraph


def nrsampling(G, size, exact=False):
    """
    Uses NRS algorithm[1] to uniformly sample connected induced subgraphs of size "size" from graph "G".

    [1] Lu X., Bressan S. (2012) Sampling Connected Induced Subgraphs Uniformly at Random.
        In: Ailamaki A., Bowers S. (eds) Scientific and Statistical Database Management.
        SSDBM 2012. Lecture Notes in Computer Science, vol 7338. Springer, Berlin, Heidelberg

    Parameters
    ----------
    G : numpy.ndarray
        Adjacency matrix of the graph. It should not contain self-loops.
    size : int
        Size of subgraph to sample
    exact : bool
        If True, the algorithm tries to find a subgraph that matches the size exactly.
        It means if the graph G does not include such subgraph, the function never returns.
        If False, the algorithm returns a subgraph that might be smaller than the required size,
        even if such subgraph exists in the graph.

    Returns
    -------
    list
        List of vertices that form the sampled subgraph.
    """
    subgraph, allowed, neighbourhood = _vxsampling(G, size, exact)
    # Fix for bias toward subgraphs with higher clustering coef.
    i = size
    neighbours = np.where(neighbourhood & allowed)[0]
    while len(neighbours) > 0:
        i += 1
        v_index = random.randint(0, len(neighbours) - 1)
        v = neighbours[v_index]
        alpha = random.random()
        if alpha < float(size) / i:
            u_index = random.randint(0, len(subgraph) - 1)

            s_prime = subgraph[:]
            s_prime[u_index] = v
            s_prime_adj = G[np.ix_(s_prime, s_prime)]
            if is_weakly_connected(s_prime_adj):
                subgraph[u_index] = v

        allowed[v] = False
        neighbourhood = neighbourhood + (G[v, :] != 0) + (G[:, v] != 0)
        neighbours = np.where(neighbourhood & allowed)[0]

    return subgraph


__all__ = ["vxsampling", "nrsampling"]
