"""Various utility functions."""
import numpy as np
import pandas as pd
from scipy.sparse import spmatrix


def gini(X: np.ndarray) -> float:
    """Calculate Gini coefficient.

    Parameters
    ----------
    X
        1D numeric (real) array.

    Returns
    -------
    gini
        Floating point number indicating the Gini coefficient of ``X``.
    """
    X = np.sort(X)
    n = X.shape[0]
    I = np.arange(1, n+1)
    return 2*np.sum(I*X) / (n*np.sum(X)) - (n+1)/n

def frustration_count(
    S: np.ndarray | spmatrix,
    gvec: np.ndarray,
    *,
    normalized: bool = True
) -> float:
    """Calculate frustration count defined
    as a sum of the (weighted) counts positive edges
    between groups and negative edges within groups.

    Parameters
    ----------
    S
        Signed adjacency matrix.
    g
        Grouping vector.
    normalized
        Should the count be normalized by the number of edges,
        so the resultin value is frustration ratio.
    """
    fidx = .0
    m    = 0
    gvec = np.array(gvec)
    for g in np.unique(gvec):
        mask = gvec == g
        X  = S[mask]
        m += np.abs(X).sum()
        x  = S[mask][:, mask]
        fidx += np.abs(x[x < 0].sum())
        x  = S[mask][:, ~mask]
        fidx += x[x > 0].sum()
    if normalized:
        fidx /= m
    return fidx

def label_clusters(gvec: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Name integer cluster identifiers using a label vector."""
    lt = pd.crosstab(labels, gvec).idxmax()
    return lt[gvec].to_numpy()
