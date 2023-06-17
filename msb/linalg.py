"""Linera algebra computations."""
from __future__ import annotations
from typing import Literal, Optional
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from scipy.special import logsumexp
from scipy.linalg import pinv, eig, eigh, eigvals, eigvalsh


_spectrum = ("LA", "SA", "BE")

def eigenstuff(
    X: np.ndarray | sp.spmatrix,
    m: int,
    *,
    symmetric: Optional[bool] = None,
    which: Literal[_spectrum] = _spectrum[0],       # type: ignore
    vectors: bool = True,
    inverse: bool = False
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Compute eigenstuff of a matrix.

    Parameters
    ----------
    X
        Matrix (sparse or array).
    m
        Number of leading eigenpairs to use.
        Twice the number if both ends of the spectrum are computed.
    symmetric
        Is the eigenproblem symmetric. Determined automatically if ``None``.
    vectors
        Should eigenvectors be computed.
    inverse
        Should (pseudo)inverse of (right) eigenvectors be returned.

    Returns
    -------
    ev
        Eigenvalues order so the upper part of the spectrum is in the front
        and the lower part is in the back.
    Q
        Corresponding eigenvectors.
    """
    # pylint: disable=too-many-branches
    if which not in _spectrum:
        raise ValueError(f"'which' has to be one of {_spectrum}")

    if symmetric is None:
        symmetric = (X != X.T).count_nonzero() == 0

    N = X.shape[0]
    if not symmetric and which == "LA":
        which = "LR"
    elif not symmetric and which == "SA":
        which = "SR"

    if m is not None and which == "BE":
        m *= 2

    if m is None or m >= N-1:
        if isinstance(X, sp.spmatrix):
            X = X.toarray()
        if vectors:
            if symmetric:
                ev = eigh(X)
            else:
                ev = eig(X, left=False, right=True)
        else:
            eigfunc = eigvalsh if symmetric else eigvals
            ev = eigfunc(X)
    elif symmetric:
        ev = splinalg.eigsh(X, k=m, which=which, return_eigenvectors=vectors)
    elif which == "BE":
        ev0 = splinalg.eigs(X, k=int(m/2), which="SR", return_eigenvectors=vectors)
        ev1 = splinalg.eigs(X, k=int(m/2), which="LR", return_eigenvectors=vectors)
        if vectors:
            Q = np.concatenate([ ev0[1], ev1[1] ], axis=1)
            ev = np.concatenate([ ev0[0], ev1[0]])
            ev = ev, Q
        else:
            ev = np.concatenate([ ev0, ev1 ])
    else:
        ev = splinalg.eigs(X, k=m, which=which, return_eigenvectors=vectors)

    if not isinstance(ev, tuple):
        ev = ev, None
    ev, Q = ev
    # Reorder eigenstuff the upper part of the spectrum
    # is in the fron and lower part in the back
    o = np.argsort(-ev)
    ev = ev[o]
    W = None
    if Q is not None:
        Q = Q[:, o]
        if inverse:
            W = pinv(Q)
    return ev, Q, W

def logmatmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    r"""Matrix multiplication (dense) in log-space.

    Parameters
    ----------
    A
        2D array.
    B
        2D array.

    Returns
    -------
    M
        Matrix such that :math:`(M)_{ij} = \log{\sum_k a_ikb_kj}`.
    """
    dtype = np.find_common_type([], [A.dtype, B.dtype])
    n = A.shape[0]
    m = B.shape[1]
    M = np.empty((n, m), dtype=dtype)
    for i in range(m):
        M[:, i] = logsumexp(A + B[None, :, i], axis=-1)
    return M
