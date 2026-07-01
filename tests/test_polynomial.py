import numpy as np

from quantum_entropy.primitives.polynomial import Polynomial


def test_constructor():

    p = Polynomial([1, 2, 3])

    assert p.degree == 2


def test_coefficients():

    p = Polynomial([1, 2, 3])

    assert np.allclose(
        p.coefficients,
        [1, 2, 3],
    )


def test_evaluation():

    p = Polynomial([1, 2, 3])

    #
    # 1 + 2x + 3x²
    #

    assert np.isclose(
        p(2),
        17,
    )


def test_derivative():

    p = Polynomial([1, 2, 3])

    dp = p.derivative()

    #
    # 2 + 6x
    #

    assert dp.degree == 1

    assert np.isclose(
        dp(2),
        14,
    )


def test_callable():

    p = Polynomial([0, 0, 1])

    assert np.isclose(
        p(5),
        25,
    )