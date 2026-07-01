import numpy as np

from quantum_entropy.core.density_operator import DensityOperator
from quantum_entropy.primitives.polynomial_transform import (
    PolynomialEigenvalueTransformation,
)


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


def test_constructor():

    pet = PolynomialEigenvalueTransformation(
        sample_density(),
        square,
    )

    assert pet.dimension == 2

    assert pet.density.dimension == 2

    assert pet.polynomial is square