import numpy as np

from quantum_entropy.core.density_operator import DensityOperator
from quantum_entropy.primitives.polynomial_transform import (
    PolynomialEigenvalueTransformation,
)

from quantum_entropy.primitives.polynomial import Polynomial



def sample_density():

    rho = np.array(
        [
            [0.7, 0.2],
            [0.2, 0.3],
        ],
        dtype=complex,
    )

    rho /= np.trace(rho)

    return DensityOperator(rho)


def square(x):
    return x * x


def sqrt_poly(x):
    return np.sqrt(x)


def test_constructor():

    pet = PolynomialEigenvalueTransformation(
        sample_density(),
        square,
    )

    assert pet.dimension == 2


def test_square():

    pet = PolynomialEigenvalueTransformation(
        sample_density(),
        square,
    )

    result = pet.apply()

    expected = sample_density().power(2)

    assert np.allclose(
        result.numpy(),
        expected,
        atol=1e-12,
    )


def test_square_root():

    pet = PolynomialEigenvalueTransformation(
        sample_density(),
        sqrt_poly,
    )

    result = pet.apply()

    expected = sample_density().sqrt()

    assert np.allclose(
        result.numpy(),
        expected,
        atol=1e-12,
    )


def test_hermitian():

    pet = PolynomialEigenvalueTransformation(
        sample_density(),
        square,
    )

    assert pet.verify_hermitian()


def test_eigenvalues():

    pet = PolynomialEigenvalueTransformation(
        sample_density(),
        square,
    )

    eigvals = sample_density().spectral_decomposition().eigenvalues

    expected = eigvals ** 2

    assert np.allclose(
        pet.transformed_eigenvalues(),
        expected,
    )

    from quantum_entropy.primitives.polynomial import Polynomial


def test_polynomial_object():

    poly = Polynomial([0, 0, 1])

    pet = PolynomialEigenvalueTransformation(
        sample_density(),
        poly,
    )

    result = pet.apply()

    expected = sample_density().power(2)

    assert np.allclose(
        result.numpy(),
        expected,
        atol=1e-12,
    )


def test_polynomial_degree():

    poly = Polynomial([1, 2, 3])

    assert poly.degree == 2