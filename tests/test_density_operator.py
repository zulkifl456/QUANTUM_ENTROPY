"""
==========================================================================
Quantum Entropy Algorithms Library

Unit Tests

Density Operator

Commit 3C

==========================================================================
"""

import numpy as np

from quantum_entropy.core.density_operator import DensityOperator


##########################################################################
# Test Matrix
##########################################################################

rho_matrix = np.array(
    [
        [0.7, 0.2],
        [0.2, 0.3]
    ],
    dtype=np.complex128
)

rho_matrix /= np.trace(rho_matrix)

rho = DensityOperator(rho_matrix)


##########################################################################
# Validation
##########################################################################

def test_is_hermitian():

    assert rho.is_hermitian()


def test_trace():

    assert np.isclose(
        rho.trace(),
        1.0
    )


def test_positive():

    assert rho.is_positive()


def test_valid():

    assert rho.is_valid()


##########################################################################
# Spectral Decomposition
##########################################################################

def test_eigenvalues():

    eig = rho.eigenvalues()

    assert len(eig) == rho.dimension

    assert np.all(eig >= -1e-12)


def test_eigenvectors():

    vec = rho.eigenvectors()

    assert vec.shape == (
        rho.dimension,
        rho.dimension
    )


def test_rank():

    assert rho.rank() == 2


##########################################################################
# Reconstruction
##########################################################################

def test_reconstruction():

    spec = rho.spectral_decomposition()

    reconstructed = spec.reconstruct()

    assert np.allclose(
        reconstructed,
        rho.matrix,
        atol=1e-12
    )


##########################################################################
# Orthonormality
##########################################################################

def test_orthonormal():

    spec = rho.spectral_decomposition()

    assert spec.is_orthonormal()


##########################################################################
# Matrix Square Root
##########################################################################

def test_square_root():

    sqrt = rho.sqrt()

    recovered = sqrt @ sqrt

    assert np.allclose(
        recovered,
        rho.matrix,
        atol=1e-10
    )


##########################################################################
# Matrix Power
##########################################################################

def test_matrix_power():

    square = rho.power(2)

    expected = rho.matrix @ rho.matrix

    assert np.allclose(
        square,
        expected,
        atol=1e-10
    )


##########################################################################
# Matrix Inverse
##########################################################################

def test_inverse():

    inv = rho.inverse()

    identity = inv @ rho.matrix

    assert np.allclose(
        identity,
        np.eye(rho.dimension),
        atol=1e-10
    )


##########################################################################
# Matrix Logarithm
##########################################################################

def test_logarithm():

    log = rho.log()

    assert log.shape == rho.shape


##########################################################################
# Cache
##########################################################################

def test_cache():

    s1 = rho.spectral_decomposition()

    s2 = rho.spectral_decomposition()

    assert s1 is s2


##########################################################################
# Copy
##########################################################################

def test_copy():

    copied = rho.copy()

    assert copied is not rho

    assert np.allclose(
        copied.matrix,
        rho.matrix
    )


##########################################################################
# NumPy Export
##########################################################################

def test_numpy():

    arr = rho.numpy()

    assert isinstance(arr, np.ndarray)

    assert np.allclose(
        arr,
        rho.matrix
    )