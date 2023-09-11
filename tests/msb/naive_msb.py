"""Naive implementation of MSB for testing purposes."""
from math import factorial
from scipy.sparse import csr_matrix, identity


def tr(X):
    """Trace operator."""
    return X.diagonal().sum()

def mpow(X, k=1, force_semi=False):
    """Matrix power."""
    if k > 2 or force_semi:
        # Semiadjacency matrix is used only for lengths k > 2
        # unless forced
        X = (X + X.T) / 2
    if k == 0:
        return identity(X.shape[0], dtype=X.dtype)
    P = X
    for _ in range(1, k):
        P = P@X
    return P

def texp(X, k0=0, k1=30):
    """Truncated exponential."""
    Y = X.copy()
    Y.data[:] = 0
    Y.eliminate_zeros()
    for k in range(k0, k1+1):
        Y += mpow(X, k) / factorial(k)
    return Y

def PNP_pow(N, P, k):
    r""":math:`PNP` matrix power used for weak balance."""
    Q = csr_matrix(N.shape, dtype=N.dtype)
    if k == 0:
        return identity(Q.shape[0], dtype=Q.dtype)
    for l in range(1, k+1):
        Q += mpow(P, l-1, force_semi=True)@N@mpow(P, k-l, force_semi=True)
    return Q

def PNP_texp(N, P, k0, k1, beta=1):
    r""":math:`PNP` truncated matrix exponential used for weak balance."""
    Q = csr_matrix(N.shape, dtype=N.dtype)
    for k in range(k0, k1+1):
        Q += PNP_pow(N, P, k) * beta**k / factorial(k)
    return Q
