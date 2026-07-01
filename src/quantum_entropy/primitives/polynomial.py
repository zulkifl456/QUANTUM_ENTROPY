"""
==========================================================================
Quantum Entropy Algorithms Library

Module
------
Polynomial

Description
-----------
Represents a real polynomial

    P(x) = c0 + c1 x + c2 x² + ...

This class provides a reusable abstraction for polynomial
evaluation and will later support

    • Polynomial Eigenvalue Transformation
    • Chebyshev Approximations
    • QSVT
    • Functional Transformations

Author
------
Zulkifl Khairoowala
==========================================================================
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


class Polynomial:
    """
    Polynomial

    Parameters
    ----------
    coefficients

        Ordered as

            [c0,c1,c2,...]
    """

    ####################################################################
    # Constructor
    ####################################################################

    def __init__(self, coefficients: Iterable[float]):

        coeffs = np.asarray(
            list(coefficients),
            dtype=float,
        )

        if coeffs.ndim != 1:
            raise ValueError(
                "Coefficients must be one-dimensional."
            )

        if coeffs.size == 0:
            raise ValueError(
                "Polynomial cannot be empty."
            )

        self._coefficients = coeffs

    ####################################################################
    # Properties
    ####################################################################

    @property
    def coefficients(self):

        return self._coefficients.copy()

    @property
    def degree(self):

        return len(self._coefficients) - 1

    ####################################################################
    # Evaluation
    ####################################################################

    def evaluate(self, x):

        return np.polynomial.polynomial.polyval(
            x,
            self._coefficients,
        )

    def __call__(self, x):

        return self.evaluate(x)

    ####################################################################
    # Utilities
    ####################################################################

    def derivative(self):

        coeffs = np.polynomial.polynomial.polyder(
            self._coefficients
        )

        return Polynomial(coeffs)

    ####################################################################
    # Representation
    ####################################################################

    def __repr__(self):

        return (
            f"Polynomial("
            f"degree={self.degree})"
        )