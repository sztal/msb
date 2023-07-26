"""Structural balance computations."""
# pylint: disable=anomalous-backslash-in-string,protected-access
# pylint: disable=too-many-public-methods,too-many-lines
# pylint: disable=no-name-in-module
from __future__ import annotations
from typing import Optional, Any, Callable
from functools import wraps, cached_property
import numpy as np
import pandas as pd
import igraph as ig
import scipy.sparse as sp
from scipy.special import logsumexp, loggamma
import joblib
from sklearn.cluster import AgglomerativeClustering
from .linalg import eigenstuff, logmatmul
from .utils import frustration_index


class Balance:
    r"""Structural balance computations.

    Attributes
    ----------
    A
        Adjacency matrix (sparse) of a graph.
        Possibly weighted. Can be passed as :py:class:`igraph.Graph`.
    directed
        Boolean indicating if graph is directed.
        Determined automatically if ``None``.
    kmin
        Minimum cycle length (adjacency matrix power) to consider
        in truncated walk balance calculations.
        If ``None`` then it is set to ``2`` for directed
        and to ``3`` for undirected networks.
    kmax
        Minimum cycle length to consider.
        The actual value is set to ``min(kmax, self.n_nodes)``,
        since no elementary cycle can be longer than
        the number of nodes.
    m
        Number of eigenvalues to use for approximating matrix powers.
    beta
        Default values of inverse temperature parameter (``beta > 0``)
        used when calculating balance measures. Inverse temperature
        works as an additional geometric weighting factor for controlling
        the characteristic length scale of the balance analysis.
        ``beta < 1`` applies more weight to shorter cycles and
        ``beta > 1`` corresponds to weighting long cycles more.
        With the proper normalization scheme for weights
        (see :py:meth:`normalize_weights`) for any network
        it can be interpreted as a multiplicative weighting factor
        applied at each step along a cycle. If ``None`` then it
        is set automatically to an appropriate range.
    semi
        Should semiwalks instead of ordinary walks.

    See Also
    --------
    find_beta_max : Finding appropriate :math:`\beta_{\text{max}}`
    """
    class Meta:
        """:py:class:`Balance` metadata and metaprogramming utilities."""
        @classmethod
        def errstate(cls, method=None, **options: str) -> Callable:
            """Temporarily modify *Numpy* warnings when running ``method``.

            ``**options`` are passed to :py:func:`numpy.errstate`.
            """
            if method is not None:
                return method
            def decorator(method):
                @wraps(method)
                def wrapped(*args: Any, **kwds: Any):
                    with np.errstate(**options):
                        return method(*args, **kwds)
                return wrapped
            return decorator

    def __init__(
        self,
        A: sp.spmatrix | ig.Graph,
        *,
        kmin: Optional[int] = None,
        kmax: int = 30,
        m: int = 5,
        beta: Optional[float] = None,
        directed: Optional[bool] = None,
        weighted: Optional[bool] = None,
        attr: Optional[str] = "weight",
        semi: bool = True,
        **kwds: Any
    ) -> None:
        """Initialization method.

        Parameters
        ----------
        attr
            Edge attribute name storing weights.
            Used only whe initializing from :py:class:`igraph.Graph`.
        **kwds
            Passed to :meth:`find_beta_max` when ``beta`` is ``None``.
        """
        self.A = A
        self.kmin = kmin
        self.kmax = kmax
        self.m = m
        self.beta = beta
        self.directed = directed
        self.weighted = weighted
        self.semi = semi
        self._cache = {}
        self.__post_init__(attr, **kwds)

    def __post_init__(self, attr: str, **kwds: Any) -> None:
        if isinstance(self.A, ig.Graph):
            self.directed = self.A.is_directed()
            edge_attrs = self.A.edge_attributes()
            if self.weighted is None:
                self.weighted = False
                if attr in edge_attrs:
                    weights = self.A.es[attr]
                    if np.unique(np.abs(weights)).size > 1:
                        self.weighted = True
            elif self.weighted and attr not in edge_attrs:
                raise AttributeError(
                    f"'weighted=True' but there is no '{attr}' edge attribute"
                )
            self.A = self.A.get_adjacency_sparse(attribute=attr)
        else:
            self.A = self.A.copy()
            if self.directed is None:
                self.directed = (self.A != self.A.T).count_nonzero() > 0
            if self.weighted is None:
                self.weighted = (np.abs(self.A.data) != 1).any()

        if self.weighted:
            self.A = self.normalize_weights(self.A, inplace=True)
        else:
            self.A = self.erase_weights(self.A, inplace=True)
        self.A = self.A.astype(float)

        if self.m is None:
            self.m = self.n_nodes

        self.set_krange(self.kmin, self.kmax)

        if self.beta is None:
            self.beta = self.find_beta_max(**kwds)

    def __repr__(self) -> str:
        cn = self.__class__.__name__
        m = self.m
        k0 = self.kmin
        k1 = self.kmax
        d = "directed" if self.directed else "undirected"
        w = "weighted" if self.weighted else "unweighted"
        n = self.n_nodes
        e = self.n_edges
        i = hex(id(self))
        return f"<{cn} with m={m} and K={k0}:{k1} on {d}, {w} network " \
            + f"with {n} nodes and {e} edges at {i}>"

    # Properties --------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return self.A.shape[0]

    @property
    def n_edges(self) -> int:
        """Number of edges."""
        return len(self.A.data)

    @property
    def strength(self) -> np.ndarray:
        """Return degree/strength sequence of the unsigned adjacency matrix."""
        return np.asarray(self.U.sum(axis=1)).squeeze()

    @property
    def S(self) -> sp.spmatrix:
        """Signed adjacency matrix."""
        return self.A

    @property
    def U(self) -> sp.spmatrix:
        """Unsigned adjacency matrix."""
        return np.abs(self.A)

    @property
    def P(self) -> sp.spmatrix:
        """Positive part of the signed adjacency matrix."""
        return self.get_P(self.A)

    @property
    def N(self) -> sp.spmatrix:
        """Negative part of the signed adjacency matrix."""
        return self.get_N(self.A)

    @cached_property
    def M(self) -> sp.spmatrix:
        r""":math:`M` matrix.

        .. math::

            M = Q^\topNQ
        """
        _, Q, W = self.eigen(self.P)
        return W@self.N@Q

    # Class & static methods --------------------------------------------------

    @staticmethod
    def normalize_weights(A: sp.spmatrix, *, inplace: bool = False) -> sp.spmatrix:
        """Normalize weights in an adjacency matrix.

        Edges are normalized so the average weight is
        equal to ``1``. This way weighted networks with
        uniform weights are equivalent to unweighted
        networks.

        Parameters
        ----------
        A
            Adjacency matrix
        inplace
            Should transformtion be done in place.

        Returns
        -------
        A
            ``A`` with normalized edge weights.
        """
        if not inplace:
            A = A.copy()
        A.data = A.data / np.abs(A.data).mean()
        return A

    @staticmethod
    def erase_weights(A: sp.spmatrix, *, inplace: bool = False) -> sp.spmatrix:
        """Erase weights from an adjacency matrix.

        Parameters
        ----------
        A
            Adjacency matrix
        inplace
            Should transformtion be done in place.

        Returns
        -------
        A
            ``A`` with only ``-1`` and ``1`` weights.
        """
        if not inplace:
            A = A.copy()
        A.data = np.sign(A.data)
        return A

    @staticmethod
    def get_P(X: sp.spmatrix) -> sp.spmatrix:
        """Get positive part of a signed adjacency matrix."""
        P = X.tocsr(copy=True)
        P.data[P.data < 0] = 0
        P.eliminate_zeros()
        return P

    @staticmethod
    def get_N(X: sp.spmatrix) -> sp.spmatrix:
        """Get negative part of a signed adjacency matrix."""
        N = X.tocsr(copy=True)
        N.data[N.data > 0] = 0
        N.eliminate_zeros()
        N.data *= -1
        return N

    # Auxiliary methods -------------------------------------------------------

    def krange(
        self,
        kmin: int = None,
        kmax: Optional[int] = None
    ) -> tuple[int, int]:
        """Get range of powers used in truncated exponentials
        with custom or default values.

        Parameters
        ----------
        kmin, kmax
            Minimum and maximum power include in the truncated power
            series. Instance attributes ``kmin`` and ``kmax`` are used
            if ``None``.
        """
        kmin = kmin if kmin is not None else self.kmin
        kmax = kmax if kmax is not None else self.kmax
        return (kmin, kmax+1)

    def B(
        self,
        bmin: float = 0,
        bmax: Optional[float] = None,
        size: int = 10
    ) -> np.ndarray:
        r"""Get vector of :math:`\beta` values.

        Parameters
        ----------
        bmin
            Minimum value. ``0`` is interpreted as positive
            machine epsilon.
        bmax
            Maximum value.
        size
            Grid size.
        """
        if bmin == 0:
            bmin = np.finfo(float).eps
        if bmax is None:
            bmax = self.beta
            if not np.isscalar(bmax):
                bmax = bmax.max()
        return np.linspace(bmin, bmax, size)

    def K(
        self,
        kmin: int = None,
        kmax: Optional[int] = None
    ) -> np.ndarray:
        """Get sequence of powers used in truncated exponentials
        with custom or default values.

        Parameters
        ----------
        kmin, kmax
            Minimum and maximum power include in the truncated power
            series. Instance attributes ``kmin`` and ``kmax`` are used
            if ``None``.
        """
        return np.arange(*self.krange(kmin, kmax))

    @staticmethod
    def make_semi(X: np.ndarray | sp.spmatrx):
        """Make semiadjacency matrix from adjacency matrix."""
        return (X + X.T) / 2

    @Meta.errstate(divide="ignore")
    def eigen(
        self,
        X: np.ndarray | sp.spmatrix,
        *,
        dropzero: bool = False,
        clog: bool = False,
        force_non_zero_ev: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return leading eigenpairs of ``X``.

        The number of leading pairs to compute is given by ``self.m``.
        Always ``self.m`` leading eigenpairs from both ends of the spectrum
        are computed. The results are cached based on :py:func:`joblib.hash`
        of ``X``. The cache is instance specific.

        Parameters
        ----------
        X
            Square matrix.
        dropzero
            Should stuff corresponding to zero eigenvalues be dropped.
        clog
            Should complex element-wise logarithm be returned.
        force_non_zero_ev
            Force zero eigenvalues to be equal
            to appropriate Numpy machine epsilon.
            If ``True`` then ``dropzero`` is ignored.

        Returns
        -------
        ev
            Eigenvalues
        Q
            Right eigenvectors.
        W
            Left eigenvectors.
        """
        if self.semi:
            X = self.make_semi(X)
        key = joblib.hash(X)
        if key not in self._cache:
            symmetric = self.semi or not self.directed
            self._cache[key] = self._calc_eigen(X, symmetric=symmetric)
        ev, Q, W = self._cache[key]
        if force_non_zero_ev:
            ev[ev == 0] = np.finfo(ev.dtype).eps
        elif dropzero:
            nz = ev != 0
            ev = ev[nz]
            Q = Q[:, nz]
            W = W[nz, :]
        if clog:
            ev, Q, W = (np.log(x.astype(complex)) for x in (ev, Q, W))
        return ev, Q, W

    @Meta.errstate(divide="ignore")
    def ltr_pow(
        self,
        X: sp.spmatrix,
        K: Optional[int | np.ndarray] = None,
    ) -> complex | np.ndarray:
        """Approximate natural logs of traces of powers of ``X``
        based on leading eigenvalues on the both sides of the spectrum.

        Parameters
        ----------
        X
            Square matrix.
        K
            1D positive integer array with powers of ``X``
            or an integer scalar.

        Returns
        -------
        Y
            Log-trace values of matrix powers as complex numbers.
            Scalar if ``K`` is a single value or an 1D array otherwise.
        """
        K = self._get_K(K)
        ev, *_ = self.eigen(X, clog=True)
        Y = logsumexp(ev*K[:, None], axis=-1)
        # Calculate zeroth power explicitly
        i = np.nonzero(K == 0)
        if i[0].size > 0:
            Y[i] = np.log(self.n_nodes)
        # Calculate first power explicitly
        i = np.nonzero(K == 1)
        if i[0].size > 0:
            diag = X.diagonal().sum()
            Y[i] = np.log(complex(diag)) if diag != 0 else complex(-np.inf, 0)
        # Calculate second power explicitly
        i = np.nonzero(K == 2)
        if i[0].size > 0:
            Y[i] = np.log(complex(X.multiply(X.T).sum()))
        return self._output(Y)

    def ltr_texp(
        self,
        X: sp.spmatrix,
        K: Optional[np.ndarray] = None,
        beta: Optional[float | np.ndarray] = None
    ) -> complex | np.ndarray:
        """Approximate natural log of trace of truncated exponential of ``X``
        based on leading eigenvalues and powers defined by ``K``.

        Parameters
        ----------
        X
            Square matrix.
        K
            1D positive integer array with powers of ``X``
            or an integer scalar.
        beta
            Inverse temperature parameter (``beta > 0``).
            Instance attribute is used when ``None``.
            See class docstring for more info.

        Returns
        -------
        Y
            Log-trace values of truncated matrix exponential as complex numbers.
            Scalar if ``T`` is a single value or an 1D array otherwise.
        """
        beta = self._get_beta(beta)
        K = self._get_K(K)
        Y = self.ltr_pow(X, K) + K*np.log(beta)[:, None] - loggamma(K+1)
        Y = logsumexp(Y, axis=-1)
        return self._output(Y)

    @Meta.errstate(divide="ignore")
    def ldiag_pow(
        self,
        X: sp.spmatrix,
        K: Optional[int | np.ndarray] = None
    ) -> np.ndarray:
        """Approximate natural logs of diagonals of powers of ``X``.

        This is approximation using truncated walk balance formalism
        based on leading eigenpairs.

        Parameters
        ----------
        X
            Square matrix.
        K
            1D positive integer array with powers of ``X``
            or an integer scalar.

        Returns
        -------
        Y
            Log-diagonal values of matrix powers as complex numbers.
            1D array with ``K`` is a single value, 2D otherwise.
        """
        K = self._get_K(K)
        ev, Q, W = self.eigen(X, clog=True)
        Y = np.empty((self.n_nodes, len(K)), dtype=complex)
        V = Q + W.T

        for i, k in enumerate(K):
            if k == 0:
                Y[:, i] = 0
            elif k == 1:
                Y[:, i] = np.log(X.diagonal().astype(complex))
            elif k == 2:
                Y[:, i] = np.log(
                    np.array(X.multiply(X.T).sum(axis=1), dtype=complex).flatten()
                )
            else:
                Y[:, i] = logsumexp(V + k*ev, axis=-1)

        return self._output(Y)

    def ldiag_texp(
        self,
        X: sp.spmatrix,
        K: Optional[np.ndarray] = None,
        beta: Optional[float | np.ndarray] = None
    ) -> complex | np.ndarray:
        """Approximate natural log of diagonal of truncated exponential of ``X``
        based on leading eigenvalues and powers from ``kmin`` to ``k``.

        Parameters
        ----------
        X
            Square matrix.
        K
            1D positive integer array with powers of ``X``
            or an integer scalar.
        beta
            Inverse temperature parameter (``beta > 0``).
            Instance attribute is used when ``None``.
            See class docstring for more info.

        Returns
        -------
        Y
            Log-diagonal values of truncated matrix exponential as complex numbers.
            1D array when ``beta`` is a single value, 2D otherwise.
        """
        beta = self._get_beta(beta)
        K = self._get_K(K)

        D = self.ldiag_pow(X, K) - loggamma(K+1)
        if D.ndim == 1:
            D = D[:, None]
        Y = np.empty((self.n_nodes, len(beta)), dtype=complex)

        for i, b in enumerate(beta):
            Y[:, i] = logsumexp(D + K*np.log(b), axis=-1)

        return self._output(Y)

    @Meta.errstate(divide="ignore")
    def lpow(
        self,
        X: sp.spmatrix,
        K: Optional[int | np.ndarray] = None
    ) -> np.ndarray:
        """Approximate element-wise complex log of power(s) of a matrix
        based on leading eigenvalues.

        Parameters
        ----------
        X
            Square matrix.
        K
            1D positive integer array with powers of ``X``
            or an integer scalar.
        """
        K = self._get_K(K)
        ev, Q, W = self.eigen(X, clog=True)
        Y = np.empty((self.n_nodes, self.n_nodes, len(K)), dtype=complex)

        for ki, k in enumerate(K):
            if k == 0:
                y = np.log(np.eye(self.n_nodes, dtype=complex))
            elif k == 1:
                y = np.log(np.array(X.todense()).astype(complex))
            elif k == 2:
                y = np.log(np.array((X@X).todense()).astype(complex))
            else:
                y = logmatmul(Q, (k*ev)[:, None] + W)
            Y[:, :, ki] = y
        return self._output(Y)

    def ltexp(
        self,
        X: sp.spmatrix,
        K: Optional[np.ndarray] = None,
        beta: Optional[float | np.ndarray] = None
    ) -> complex | np.ndarray:
        """Approximate element-wise complex natural log of
        a matrix exponential based on leading eigenvalues.

        Parameters
        ----------
        X
            Square matrix.
        K
            1D positive integer array with powers of ``X``
            or an integer scalar.
        beta
            Inverse temperature parameter (``beta > 0``).
            Instance attribute is used when ``None``.
            See class docstring for more info.

        Returns
        -------
        Y
            Dense 2D array with log values of the matrix
            exponential of ``X``. If more than one value of ``beta``
            then a 3D array is returned with the last axis keeping
            track of ``beta`` values.
        """
        beta = self._get_beta(beta)
        K = self._get_K(K)
        Y = np.empty((self.n_nodes, self.n_nodes, len(beta)), dtype=complex)

        for bi, b in enumerate(beta):
            y = (K*np.log(b) - loggamma(K+1))[None, None, :] + self.lpow(X, K)
            Y[:, :, bi] = logsumexp(y, axis=-1)
        return self._output(Y)

    @Meta.errstate(divide="ignore")
    def lnL(self, k: int) -> np.ndarray:
        r"""Get element-wise log of :math:`L` matrix.

        .. math::

            L(k) = \sum_{l=1}^k L(k,j)

        .. math::

            L(k,l)_{ij} = \lambda_i^{l-1}\lambda_j^{k-l}
        """
        ev, *_ = self.eigen(self.P, clog=True)
        l = np.arange(1, k+1)
        L1 = ev[:, None]*(l-1)
        L2 = ev[:, None]*(k-l)
        L  = logsumexp(L1[:, None, :] + L2, axis=-1)
        return self._output(L)

    @Meta.errstate(divide="ignore")
    def ltr_Vk(
        self,
        K: Optional[int | np.ndarray] = None
    ) -> complex | np.ndarray:
        """Approximate natural log of the trace of :math:`V_k` matrix
        based on leading eigenvalues.

        Parameters
        ----------
        K
            1D positive integer array with powers of ``X``
            or an integer scalar.
        """
        K = self._get_K(K)
        M = np.log(self.M.diagonal().astype(complex))
        ev, *_ = self.eigen(self.P, clog=True)
        Y = np.log(K) + logsumexp(ev[:, None]*(K-1) + M[:, None], axis=0)
        return self._output(Y)

    @Meta.errstate(divide="ignore")
    def ltr_V(
        self,
        K: Optional[int | np.ndarray] = None,
        beta: Optional[float | np.ndarray] = None
    ) -> complex | np.ndarray:
        """Approximate natural logs of :math:`V` matrix
        based on leading eigenvalues.

        Parameters
        ----------
        K
            1D positive integer array with powers of ``X``
            or an integer scalar.
        beta
            Inverse temperature parameter (``beta > 0``).
            Instance attribute is used when ``None``.
            See class docstring for more info.
        """
        beta = self._get_beta(beta)
        K = self._get_K(K)
        M = np.log(self.M.astype(complex))
        ev, *_ = self.eigen(self.P, clog=True)

        X1 = K[:, None]*np.log(beta) - loggamma(K)[:, None]
        X2 = logsumexp(ev[:, None]*(K-1) + M.diagonal()[:, None], axis=0)
        Y  = logsumexp(X1 + X2[:, None], axis=0)
        return self._output(Y)

    @Meta.errstate(divide="ignore")
    def _lnUk(
        self,
        K: Optional[int | np.ndarray] = None
    ) -> np.ndarray:
        """Inner term in :math:`V_k` matrix.

        Parameters
        ----------
        K
            1D positive integer array with powers of ``X``
            or an integer scalar.
        """
        K = self._get_K(K)
        ev, *_ = self.eigen(self.P, clog=True)
        M = np.log(self.M.astype(complex))
        U = np.empty((len(ev), len(ev), len(K)), dtype=complex)
        for ki, k in enumerate(K):
            U[:, :, ki] = self.lnL(k)
        if U.ndim > M.ndim:
            M = M[:, :, None]
        U += M
        return U

    @Meta.errstate(divide="ignore")
    def ldiag_Vk(
        self,
        K: Optional[int | np.ndarray] = None
    ) -> np.ndarray:
        """Approximate diagonal powers of :math:`V_k` matrix
        based on leading eigenvalues.

        Parameters
        ----------
        K
            1D positive integer array with powers of ``X``
            or an integer scalar.
        """
        K = self._get_K(K)
        U = self._lnUk(K)
        _, Q, W = self.eigen(self.P, clog=True)
        V = np.empty((U.shape[0], U.shape[-1]), dtype=complex)
        for ki in range(len(K)):
            V[:, ki] = logsumexp(logmatmul(Q, U[:, :, ki]) + W.T, axis=-1)
        return self._output(V)

    @Meta.errstate(divide="ignore")
    def lnVk(
        self,
        K: Optional[int | np.ndarray] = None
    ) -> np.ndarray:
        """Approximate element-wise log of :math:`V_k` matrix
        based on leading eigenvalues.

        Parameters
        ----------
        K
            1D positive integer array with powers of ``X``
            or an integer scalar.
        """
        K = self._get_K(K)
        U = self._lnUk(K)
        V = np.empty((self.n_nodes, self.n_nodes, len(K)), dtype=complex)
        _, Q, W = self.eigen(self.P, clog=True)
        for ki in range(len(K)):
            V[:, :, ki] = logmatmul(logmatmul(Q, U[:, :, ki]), W)
        return self._output(V)

    @Meta.errstate(divide="ignore")
    def _lnU(
        self,
        K: Optional[int | np.ndarray] = None,
        beta: Optional[float | np.ndarray] = None
    ) -> np.ndarray:
        """Approximate element-wise log of :math:`U` matrix
        (inner part of :math:`V`).

        Parameters
        ----------
        K
            1D positive integer array with powers of ``X``
            or an integer scalar.
        beta
            Inverse temperature parameter (``beta > 0``).
            Instance attribute is used when ``None``.
            See class docstring for more info.
        """
        K = self._get_K(K)
        beta = self._get_beta(beta)
        ev, *_ = self.eigen(self.P, clog=True)
        M = np.log(self.M.astype(complex))
        U = np.empty((len(ev), len(ev), len(beta), len(K)), dtype=complex)
        for bi, b in enumerate(beta):
            for ki, k in enumerate(K):
                U[:, :, bi, ki] = k*np.log(b) - loggamma(k+1) + self.lnL(k)
        U = logsumexp(U, axis=-1)
        if U.ndim > M.ndim:
            M = M[:, :, None]
        U += M
        return U

    @Meta.errstate(divide="ignore")
    def ldiag_V(
        self,
        K: Optional[int | np.ndarray] = None,
        beta: Optional[float | np.ndarray] = None
    ) -> np.ndarray:
        """Approximate element-wise log of the diagonal of :math:`V`
        matrix based on leading eigenvalues.

        Parameters
        ----------
        K
            1D positive integer array with powers of ``X``
            or an integer scalar.
        beta
            Inverse temperature parameter (``beta > 0``).
            Instance attribute is used when ``None``.
            See class docstring for more info.
        """
        beta = self._get_beta(beta)
        _, Q, W = self.eigen(self.P, clog=True)
        U = self._lnU(K, beta)
        V = np.empty((self.n_nodes, len(beta)), dtype=complex)
        for bi in range(len(beta)):
            V[:, bi] = logsumexp(logmatmul(Q, U[:, :, bi]) + W.T, axis=-1)
        return self._output(V)

    @Meta.errstate(divide="ignore")
    def lnV(
        self,
        K: Optional[int | np.ndarray] = None,
        beta: Optional[float | np.ndarray] = None
    ) -> np.ndarray:
        """Approximate element-wise log of :math:`V` matrix
        based on leading eigenvalues.

        Parameters
        ----------
        K
            1D positive integer array with powers of ``X``
            or an integer scalar.
        beta
            Inverse temperature parameter (``beta > 0``).
            Instance attribute is used when ``None``.
            See class docstring for more info.
        """
        beta = self._get_beta(beta)
        _, Q, W = self.eigen(self.P, clog=True)
        U = self._lnU(K, beta)
        V = np.empty((self.n_nodes, self.n_nodes, len(beta)), dtype=complex)
        for bi in range(len(beta)):
            V[:, :, bi] = logmatmul(logmatmul(Q, U[:, :, bi]), W)
        return self._output(V)

    # Main methods ------------------------------------------------------------

    def balance_index(
        self,
        beta: Optional[float | np.ndarray] = None,
        *,
        weak: bool = False,
        **kwds: Any
    ) -> float | np.ndarray:
        r"""Calculate balane index (strong or weak).

        It is approximated using truncated walk-balance formalism
        and based on leading eigenvalues. In the strong case it is
        defined as:

        .. math::

            \frac{\mu_B - \mu_U}{\mu_B + \mu_U}

        And in the weak case as:

        .. math::

            1 - \frac{\mu_W}{\mu_B + \mu_U}

        Parameters
        ----------
        beta
            Inverse temperature parameter (``beta > 0``).
            Instance attribute is used when ``None``.
            See class docstring for more info.
        weak
            Should weak balance be calculated.
        **kwds
            Passed to :py:meth:`ltr_texp`
            and :py:meth:`ltr_V` (if ``weak=True``).

        Returns
        -------
        K
            Balance index for different values of ``beta``.
        """
        beta = self._get_beta(beta)
        U = self.ltr_texp(self.U, beta=beta, **kwds)
        if weak:
            V = self.ltr_V(beta=beta, **kwds)
            S = self._S_transform(V, U)
        else:
            S = self.ltr_texp(self.S, beta=beta, **kwds)

        K = np.real(np.exp(S - U))
        if np.isscalar(K):
            return K
        name = "weak" if weak else "strong"
        beta = pd.Series(beta, name="beta")
        return pd.Series(K, index=beta, name=name)

    def balance(
        self,
        beta: Optional[float | np.ndarray] = None,
        *,
        weak: bool = False,
        **kwds: Any
    ) -> float | np.ndarray:
        r"""Calculate degree of balance (strong or weak).

        In both cases it is defined as the fraction of balanced
        closed walks. In both strong and weak case the degree of balance
        is defined in terms of the corresponding (strong or weak)
        balance index (:math:`K_S` or :math:`K_W`) as:

        .. math::

            \frac{K + 1}{2}

        Parameters
        ----------
        beta
            Inverse temperature parameter (``beta > 0``).
            Instance attribute is used when ``None``.
            See class docstring for more info.
        weak
            Should weak balance be calculated.
        **kwds
            Passed to :py:meth:`ltr_texp`
            and :py:meth:`ltr_NPk_texp` (if ``weak=True``).

        Returns
        -------
        B
            Walk-balance measure as a float
            or a 1D :py:class:`pandas.Series` if ``T`` is array-like.
        """
        beta = self._get_beta(beta)
        J = self.balance_index(beta, weak=weak, **kwds)
        B = (J + 1) / 2
        Y = self._output(B)
        if isinstance(Y, np.ndarray):
            name = "weak" if weak else "strong"
            if not np.isscalar(beta):
                beta = pd.Series(beta, name="beta")
            Y = pd.Series(Y, index=beta, name=name)
        return Y

    def contrib(
        self,
        beta: Optional[float | np.ndarray] = None,
        *,
        extra_kmax: int = 0,
        K: Optional[np.ndarray] = None,
    ) -> pd.Series:
        r"""Contribution scores for cycles of different lengths.

        Parameters
        ----------
        beta
            Inverse temperature parameter (``beta > 0``).
            Instance attribute is used when ``None``.
            See class docstring for more info.
        extra_kmax
            Number of additional powers to include when calculating
            the denominator in contribution calculations.
            Should be non-negative. When set to positive values it can
            be useful for diagnosing whether a given value of ``kmax``
            is large enough for accurate approximations.
        K
            Sequence of cycle lengths to consider.

        Returns
        -------
        C
            Series with contribution scores
            indexed by ``beta`` values and cycle lengths.
        """
        # pylint: disable=too-many-locals
        beta = self._get_beta(beta)
        K = self._get_K(K)
        U = self.ltr_pow(self.U, K)

        # Calculate contribution profile(s)
        C = U + K*np.log(beta)[:, None] - loggamma(K+1)
        if extra_kmax > 0:
            K2 = np.arange(K.min(), K.max()+extra_kmax)
        else:
            K2 = K
        D = self.ltr_texp(self.U, K=K2, beta=beta)
        C = np.real(np.exp(C.T - D))
        C = self._output(C)

        C = pd.DataFrame(
            data=C,
            index=pd.Series(K, name="K"),
            columns=pd.Series(beta, name="beta")
        ).unstack()
        C.name = "contrib"

        if extra_kmax <= 0:
            C = C.groupby(level="beta").transform(lambda x: x / x.sum())

        return C

    def k_balance(
        self,
        beta: Optional[float | np.ndarray] = None,
        *,
        weak: bool = False,
        K: Optional[np.ndarray] = None
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """:math:`k`-balance profile.

        Parameters
        ----------
        beta
            Inverse temperature parameter (``beta > 0``).
            Instance attribute is used when ``None``.
            See class docstring for more info.
        weak
            Should weak balance be calculated.
        K
            Sequence of cycle lengths to consider.

        Returns
        -------
        B
            Balance scores for different cycle lenghts
            indexed by ``beta`` values.
        """
        # pylint: disable=too-many-locals
        beta = self._get_beta(beta)
        K = self._get_K(K)
        U = self.ltr_pow(self.U, K)

        # Calculate balance profile
        if weak:
            V = self.ltr_Vk(K)
            S = self._S_transform(V, U)
        else:
            S = self.ltr_pow(self.S, K)
        J = np.real(np.exp(S - U))
        B = (J + 1) / 2
        B = self._output(B)
        name = "weak" if weak else "strong"
        K = pd.Series(K, name="K")
        B = pd.Series(B, index=K, name=name)
        return B

    @Meta.errstate(invalid="ignore")
    def node_balance(
        self,
        beta: Optional[float | np.ndarray] = None,
        *,
        weak: bool = False,
        clip: bool = True,
        report_incorrect: bool = False,
        nans_as_zeros: bool = False,
        **kwds: Any
    ) -> np.ndarray:
        """Node-level balance measures.

        This is an approximation using truncated walk-balance formalism
        based on leading eigenvalues.

        Parameters
        ----------
        beta
            Inverse temperature parameter (``beta > 0``).
            Instance attribute is used when ``None``.
            See class docstring for more info.
        weak
            Should weak balance be calculated.
        clip
            Should values be clipped to be between ``0`` and ``1``.
            This is useful as the approximation may sometimes generate
            values ooutside of this range.
        report_incorrect
            Return additionaly fraction of values which were originally
            (that is, before clipping) out of ``[0, 1]`` bounds.
             It gives a rough estimate of how far off
            results may be due to approximation error.
        **kwds
            Passed to :py:meth:`ldiag_texp` and :py:meth:`ldiag_NPk_texp`
            (if ``weak=True``).

        Returns
        -------
        B
            Array with walk-balance measures for individual nodes.
        e
            Rough estimate of the fraction of erroneous values.
        """
        # pylint: disable=too-many-locals
        beta = self._get_beta(beta)
        U = self.ldiag_texp(self.U, beta=beta, **kwds)
        if weak:
            V = self.ldiag_V(beta=beta, **kwds)
            S = self._S_transform(V, U)
        else:
            S = self.ldiag_texp(self.S, beta=beta, **kwds)

        J = np.real(np.exp(S - U))
        B = (1 + J) / 2

        if nans_as_zeros:
            B[np.isnan(B)] = 0
        if report_incorrect:
            e = np.nanmean((B < 0) | (B > 1))
        if clip:
            B = np.clip(B, 0, 1)

        B = pd.DataFrame(B, columns=pd.Series(beta, name="beta")) \
            .unstack() \
            .swaplevel() \
            .sort_index()
        name = "weak" if weak else "strong"
        B.name = name

        if report_incorrect:
            return B, e
        return B

    @Meta.errstate(invalid="ignore")
    def node_contrib(
        self,
        beta: Optional[float | np.ndarray] = None,
        *,
        normalize: bool = False,
        **kwds: Any
    ) -> np.ndarray:
        r"""Node-level contributions.

        Parameters
        ----------
        beta
            Inverse temperature parameter (``beta > 0``).
            Instance attribute is used when ``None``.
            See class docstring for more info.
        normalize
            Should contributions be normalized so they
            sum up to ``self.n_nodes``. This makes it easier
            to compare values between nodes in different networks.
        **kwds
            Passed to :py:meth:`ldiag_texp`.

        Returns
        -------
        C
            1D float array with contributions of individual nodes.
        """
        beta = self._get_beta(beta)
        U = self.ldiag_texp(self.U, beta=beta, **kwds)
        C = np.real(np.exp(U - logsumexp(U, axis=0)))
        C = pd.DataFrame(C, columns=pd.Series(beta, name="beta")) \
            .unstack() \
            .swaplevel() \
            .sort_index()
        C.name = "contrib"

        if normalize:
            C *= self.n_nodes

        return C

    def pairwise_index(
        self,
        *,
        weak: bool = False,
        kmin: int = 1,
        **kwds: Any
    ) -> np.ndarray:
        """Calculate pairwise cohesion index.

        Parameters
        ----------
        weak
            Should weak balance be used.
        kmin
            Minimum walk length to consider.
            Typically should be ``1``.
        **kwds
            Passed to :meth:`ltexp`
            (and :meth:`lnV` when ``weak=True``).
        """
        kwds = dict(K=self.K(kmin), **kwds) # pylint: disable=use-dict-literal
        U = self.ltexp(self.U, **kwds)
        if weak:
            V = self.lnV(**kwds)
            S = self._S_transform(V, U)
        else:
            S = self.ltexp(self.S, **kwds)
        J = np.real(np.exp(S - U))
        return J

    def pairwise_cohesion(self, **kwds: Any) -> np.ndarray:
        """Pairwise degree of cohesion.

        Parameters
        ----------
        **kwds
            Passed to :meth:`pairwise_index`.
        """
        J = self.pairwise_index(**kwds)
        return (J + 1) / 2

    def find_clusters(
        self,
        beta: Optional[float] = None,
        *,
        clust_kws: Optional[dict] = None,
        full_results: bool = False,
        min_clusters: int = 2,
        max_clusters: int = 10,
        **kwds: Any
    ) -> pd.DataFrame | np.ndarray:
        """Find clusters that minimize frustration index.

        Parameters
        ----------
        beta
            Inverse temperature to use. Only a single scalar value
            can be passed. ``self.beta.max()`` is used when ``None``.
        clust_kws
            Mapping with additional arguments for the clustering method
            (``affinity`` argument is ignored).
        full_results
            Should full results for all partitions be returned
            instead of only the best one.
        max_clusters
            Maximum number of clusters to consider.
        **kwds
            Passed to :meth:`pairwise_cohesion`
            (``weak`` argument is ignored).

        Returns
        -------
        fidx, hc
            Frustration index and the best clustering solution.
        data
            Data frame with all clustering solutions.
            Returned when ``full_results=True``.
        """
        # pylint: disable=too-many-locals
        if beta is None:
            beta = self.beta
            if not np.isscalar(beta):
                beta = beta.max()
        clust_kws = {
            "linkage": "average",
            **(clust_kws or {}),
            "metric": "precomputed"
        }
        N     = np.arange(min_clusters, min(max_clusters, self.n_nodes) + 1)
        data  = dict(n=N)
        modes = ("s", "w")

        for mode in modes:
            kwds["weak"] = mode == "w"
            dist = 1 - self.pairwise_cohesion(beta=beta, **kwds)
            np.fill_diagonal(dist, 0)
            for n in N:
                hc = AgglomerativeClustering(n_clusters=n, **clust_kws)
                hc.fit(dist)
                fidx = frustration_index(self.S, hc.labels_)
                data.setdefault("fidx_"+mode, []).append(fidx)
                data.setdefault("hc_"+mode, []).append(hc)

        data = pd.DataFrame(data)
        if full_results:
            return data
        cols = [ "fidx_"+m for m in modes ]
        fcol = data[cols].min().idxmin()
        mode = fcol.split("_")[-1]
        idx  = data[fcol].idxmin()
        fidx, hc = data.loc[idx, ["fidx_"+mode, "hc_"+mode]]
        return fidx, hc

    # Auxiliary methods -------------------------------------------------------

    def set_krange(
        self,
        kmin: Optional[int] = None,
        kmax: Optional[int] = None
    ) -> None:
        r"""Set :math:`k_{\min}` and/or :math:`k_{\max}`."""
        for name, k in dict(kmin=kmin, kmax=kmax).items():
            if k is None:
                continue
            if not isinstance(k, (int, np.integer)):
                raise TypeError(f"'{name}' must be an integer")
            if k < 0:
                raise ValueError(f"'{name}' must be positive")
        kmax = min(kmax, self.n_nodes)
        if kmin is None:
            kmin = 2 if self.directed else 3
        if kmax < kmin:
            raise ValueError("'kmax' cannot be lower than 'kmin'")
        self.kmin = int(kmin)
        self.kmax = int(kmax)

    def find_beta_max(
        self,
        *,
        tol: float = 1e-6,
        search_beta_min: float = 1e-9,
        search_beta_max: float = 10,
        search_grid_size: int = 100,
        max_iter: int = 100,
        alpha: float = 1,
        **kwds: Any
    ) -> float:
        r"""Find :math:`\beta_{\max}`.

        Parameters
        ----------
        tol
            Numerical tolerance for checking whether
            :math:`\beta` has significantly changed.
        search_beta_max
            Maximum value of :math:`\beta` during
            the initial grid search.
        search_grid_size
            Number of points used during grid search.
        max_iter
            Maximum search iterations.
        alpha
            The cumulative fraction of contribution profile
            to consider when determining monotonicity.
        **kwds
            Keyword arguments other than ``beta``
            passed to :meth:`contrib`.

        Returns
        -------
        beta_max
            :math:`\beta_{\text{max}}` up to ``tol`` accuracy.

        Raises
        ------
        StopIteration
            When ``max_iter`` is reached before
            finding :math:`\beta_{\max}`.
        """
        # pylint: disable=too-many-locals
        beta_max = np.inf
        start = search_beta_min
        end = search_beta_max
        niter = 0
        while True:
            beta = np.linspace(start, end, search_grid_size)
            C = self.contrib(beta, **kwds)
            seq = C.groupby(level="beta", group_keys=False) \
                .apply(lambda x: x[x.cumsum().shift(fill_value=0) < alpha]) \
                .groupby(level="beta") \
                .is_monotonic_decreasing
            start = seq[::-1].idxmax()
            if start == beta.max():
                end = start*2
            else:
                end = start + (beta.max()-beta.min()) / (search_grid_size-1)
            if np.abs(beta_max - start) <= tol:
                return start
            beta_max = start
            niter += 1
            if niter >= max_iter:
                raise StopIteration("'max_iter' reached before finding 'beta_max'")

    # Internals ---------------------------------------------------------------

    def _calc_eigen(self, X, symmetric):
        return eigenstuff(
            X=X,
            m=self.m,
            symmetric=symmetric,
            which="BE",
            vectors=True,
            inverse=True
        )

    @staticmethod
    def _input(X, dtype=None):
        if np.isscalar(X):
            X = np.array([X])
        elif not isinstance(X, np.ndarray):
            X = np.array(X)
        if dtype is not None:
            X = X.astype(dtype)
        return X

    @staticmethod
    def _output(X, dtype=None):
        X = X.squeeze()
        if X.size == 1:
            X = X.sum()
        if dtype is not None:
            X = X.astype(dtype)
        return X

    def _get_beta(self, beta):
        if beta is None:
            beta = self.beta
        beta = self._input(beta)
        return beta

    def _get_K(self, K, *args, **kwds):
        if K is None:
            K = self.K(*args, **kwds)
        K = self._input(K)
        return K

    def _ladd(self, *args):
        tup = tuple(self._input(x) for x in args)
        return logsumexp(np.stack(tup, axis=-1), axis=-1)

    @staticmethod
    def _S_transform(V, U):
        return U + np.log(1 - 2*np.exp(V - U))
