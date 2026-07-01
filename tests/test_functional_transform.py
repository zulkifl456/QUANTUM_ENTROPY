import numpy as np

from quantum_entropy.core.density_operator import DensityOperator
from quantum_entropy.primitives.functional_transform import (
    FunctionalTransformation,
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


def square():

    return Polynomial([0, 0, 1])


def test_constructor():

    ft = FunctionalTransformation(
        sample_density(),
        square,
    )

    assert ft.dimension == 2


def test_apply():

    ft = FunctionalTransformation(
        sample_density(),
        square,
    )

    result = ft.apply()

    expected = sample_density().power(2)

    assert np.allclose(
        result.numpy(),
        expected,
        atol=1e-12,
    )