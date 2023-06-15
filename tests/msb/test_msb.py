"""Main tests for the MSB implementation.

This test suite replicates all the tests from ``2-tests.ipynb`` notebook.
Its purpose is to provide a way to run these tests automatically
using :mod:`pytest`.
"""
# pylint: disable=redefined-outer-name
from pathlib import Path
import pytest
import igraph as ig
import numpy as np
from scipy.sparse import csr_array
from msb import Balance
from .naive_msb import tr, mpow, texp, PNP_pow, PNP_texp


@pytest.fixture(scope="session")
def B() -> Balance:
    """Balance object used in tests."""
    root = Path(__file__).absolute().parent.parent.parent
    data = root/"data"/"sampson"
    G = ig.Graph.Read_GraphMLz(str(data/"t4.graphml.gz"))
    B = Balance(G, beta=[1/3, 1/2, 1], m=None, kmax=30)
    return B


class TestTracePowers:
    """Test trace power calculations."""

    def test_unsigned_adjacency_matrix(self, B) -> None:
        """Test unsigned adjacency matrix."""
        X = B.U
        ## CALCULATED VALUES
        Y = np.real(np.exp(B.ltr_pow(X)))
        ## EXPECTED VALUE
        E = np.array([ tr(mpow(X, k)) for k in B.K() ])
        assert np.allclose(Y, E)

    def test_signed_adjacency_matrix(self, B) -> None:
        """Test signed adjacency matrix."""
        X = B.S
        ## CALCULATED VALUES
        Y = np.real(np.exp(B.ltr_pow(X)))
        ## EXPECTED VALUE
        E = np.array([ tr(mpow(X, k)) for k in B.K() ])
        assert np.allclose(Y, E)


class TestTraceExponentials:
    """Test trace exponential calculations."""

    def test_unsigned_adjacency_matrix(self, B) -> None:
        """Unsigned adjacency matrix."""
        X = B.U
        ## CALCULATED VALUES
        Y = np.real(np.exp(B.ltr_texp(X, beta=B.beta, K=B.K(0))))
        ## EXPECTED VALUES
        E = np.array([ tr(texp(X*b)) for b in B.beta ])
        assert np.allclose(Y, E, rtol=1e-4)

    def test_signed_adjacency_matrix(self, B) -> None:
        """Signed adjacency matrix."""
        X = B.S
        ## CALCULATED VALUES
        Y = np.real(np.exp(B.ltr_texp(X, beta=B.beta, K=B.K(0))))
        ## EXPECTED VALUES
        E = np.array([ tr(texp(X*b)) for b in B.beta ])
        assert np.allclose(Y, E)


class TestDiagonalPowers:
    """Test diagonal powers calculations."""

    def test_unsigned_adjacency_matrix(self, B) -> None:
        """Unsigned adjacency matrix."""
        X = B.U
        ## CALCULATED VALUES
        Y = np.real(np.exp(B.ldiag_pow(X)))
        ## EXPECTED VALUES
        E = np.column_stack([ mpow(X, k).diagonal() for k in B.K() ])
        assert np.allclose(Y, E)

    def test_signed_adjacency_matrix(self, B) -> None:
        """Signed adjacency matrix."""
        X = B.S
        ## CALCULATED VALUES
        Y = np.real(np.exp(B.ldiag_pow(X)))
        ## EXPECTED VALUES
        E = np.column_stack([ mpow(X, k).diagonal() for k in B.K() ])
        assert np.allclose(Y, E)


class TestDiagonalExponentials:
    """Test diagonal exponentials calculations."""

    def test_unsigned_adjacency_matrix(self, B) -> None:
        """Unsigned adjacency matrix."""
        X = B.U
        ## CALCULATED VALUES
        Y = np.real(np.exp(B.ldiag_texp(X, beta=B.beta, K=B.K(0))))
        ## EXPECTED VALUES
        E = np.column_stack([ texp(X*b).diagonal() for b in B.beta ])
        assert np.allclose(Y, E, rtol=1e-4)

    def test_signed_adjacency_matrix(self, B) -> None:
        """Signed adjacency matrix."""
        X = B.S
        ## CALCULATED VALUES
        Y = np.real(np.exp(B.ldiag_texp(X, beta=B.beta, K=B.K(0))))
        ## EXPECTED VALUES
        E = np.column_stack([ texp(X*b).diagonal() for b in B.beta ])
        assert np.allclose(Y, E)


class TestMatrixPowers:
    """Test matrix powers calculations."""

    def test_unsigned_adjacency_matrix(self, B) -> None:
        """Unsigned adjacency matrix."""
        X = B.U
        for k in B.K():
            Y = np.real(np.exp(B.lpow(X, k)))
            E = csr_array(mpow(X, k)).todense()
            assert np.allclose(Y, E)

    def test_signed_adjacency_matrix(self, B) -> None:
        """Signed adjacency matrix."""
        X = B.S
        for k in B.K():
            Y = np.real(np.exp(B.lpow(X, k)))
            E = csr_array(mpow(X, k)).todense()
            assert np.allclose(Y, E)


class TestMatrixExponentials:
    """Test matrix exponentials calculations."""

    def test_unsigned_adjacency_matrix(self, B) -> None:
        """Unsigned adjacency matrix."""
        X = B.U
        for beta in B.beta:
            Y = np.real(np.exp(B.ltexp(X, beta=beta)))
            E = csr_array(texp(X*beta, *B.krange())).todense()
            assert np.allclose(Y, E, rtol=1e-4)

    def test_signed_adjacency_matrix(self, B) -> None:
        """Signed adjacency matrix."""
        X = B.S
        for beta in B.beta:
            Y = np.real(np.exp(B.ltexp(X, beta=beta)))
            E = csr_array(texp(X*beta, *B.krange())).todense()
            assert np.allclose(Y, E, rtol=1e-2)


class TestPNP:
    r"""Test :math:`PNP` calculations."""

    def test_trace_powers(self, B) -> None:
        r"""Test :math:`PNP` trace powers calculations."""
        N = B.N
        P = B.P
        ## CALCULATED VALUES
        Y = np.real(np.exp(B.ltr_Vk()))
        ## EXPECTED VALUES
        E = np.array([ tr(PNP_pow(N, P, k)) for k in B.K() ])
        assert np.allclose(Y, E)

    def test_trace_truncated_exponentials(self, B) -> None:
        r"""Test :math:`PNP` trace truncated exponentials calculations."""
        N = B.N
        P = B.P
        ## CALCULATED VALUES
        Y = np.real(np.exp(B.ltr_V(beta=B.beta)))
        ## EXPECTED VALUES
        E = np.array([ tr(PNP_texp(N, P, *B.krange(), beta=b)) for b in B.beta ])
        assert np.allclose(Y, E)

    def test_diagonal_powers(self, B) -> None:
        r"""Test :math:`PNP` diagonal powers calculations."""
        N = B.N
        P = B.P
        ## CALCULATED VALUES
        Y = np.real(np.exp(B.ldiag_Vk()))
        ## EXPECTED VALUES
        E = np.column_stack([ PNP_pow(N, P, k).diagonal() for k in B.K() ])
        assert np.allclose(Y, E)

    def test_diagonal_truncated_exponentials(self, B) -> None:
        r"""Test :math:`PNP` diagonal truncated exponentials calculations."""
        N = B.N
        P = B.P
        ## CALCULATED VALUES
        Y = np.real(np.exp(B.ldiag_V()))
        ## EXPECTED VALUES
        E = np.column_stack([ PNP_texp(N, P, *B.krange(), beta=b).diagonal() for b in B.beta ])
        assert np.allclose(Y, E)

    def test_matrix_powers(self, B) -> None:
        r"""Test :math:`PNP` matrix powers calculations."""
        N = B.N
        P = B.P
        for k in B.K():
            Y = np.real(np.exp(B.lnVk(k)))
            E = csr_array(PNP_pow(N, P, k)).todense()
            assert np.allclose(Y, E)

    def test_matrix_exponentials(self, B) -> None:
        r"""Test :math:`PNP` matrix exponentials calculations."""
        N = B.N
        P = B.P
        for beta in B.beta:
            Y = np.real(np.exp(B.lnV(beta=beta)))
            E = csr_array(PNP_texp(N, P, *B.krange(), beta=beta)).todense()
            assert np.allclose(Y, E)
