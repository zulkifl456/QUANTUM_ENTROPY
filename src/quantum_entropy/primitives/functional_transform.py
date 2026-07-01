"""
==========================================================================
Quantum Entropy Algorithms Library

Module
------
Functional Transformation

Description
-----------
Approximates matrix-valued functions

    f(A)

using polynomial approximations.

This class implements the numerical analogue of
Theorem II.5.

==========================================================================
"""

from __future__ import annotations

from typing import Callable

from quantum_entropy.primitives.matrix_function import MatrixFunction
from quantum_entropy.primitives.polynomial_transform import (
    PolynomialEigenvalueTransformation,
)


class FunctionalTransformation(MatrixFunction):
    """
    Matrix functional transformation.

    Parameters
    ----------
    density

    approximation

        Callable returning a Polynomial approximation.
    """

    def __init__(

        self,

        density,

        approximation: Callable,

    ):

        super().__init__(density)

        if not callable(approximation):

            raise TypeError(
                "Expected callable approximation."
            )

        self._approximation = approximation

    @property
    def approximation(self):

        return self._approximation

    def apply(self):

        if self._result is not None:
            return self._result

        polynomial = self._approximation()

        transform = PolynomialEigenvalueTransformation(
            self.density,
            polynomial,
        )

        self._result = transform.apply()

        return self._result

    def __repr__(self):

        return (
            "FunctionalTransformation("
            f"dimension={self.dimension})"
        )