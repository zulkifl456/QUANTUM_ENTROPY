"""
==========================================================================
Quantum Entropy Algorithms Library

Module
------
Polynomial Eigenvalue Transformation

Description
-----------
Implements the numerical analogue of the Polynomial Eigenvalue
Transformation introduced in Section II of

    New Quantum Algorithms for Computing Quantum Entropies
    and Distances of Density Operators

A polynomial

    P(x)

is applied to the eigenvalues of a density operator while preserving
its eigenvectors.

This is the numerical counterpart of the quantum polynomial
transformation used later for

    • Positive Powers
    • Matrix Logarithm
    • Von Neumann Entropy
    • Rényi Entropy
    • Trace Distance
    • Fidelity

Author
------
Zulkifl Khairoowala
==========================================================================
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from quantum_entropy.primitives.matrix_function import MatrixFunction
from quantum_entropy.primitives.polynomial import Polynomial


class PolynomialEigenvalueTransformation(MatrixFunction):
    """
    Numerical Polynomial Eigenvalue Transformation.
    """

    ####################################################################
    # Constructor
    ####################################################################

    def __init__(
        self,
        density,
        polynomial,
    ):

        super().__init__(density)

        if isinstance(polynomial, Polynomial):

            self._polynomial = polynomial

        elif callable(polynomial):

            self._polynomial = polynomial

        else:

            raise TypeError(
                "Expected Polynomial or callable."
            )

    ####################################################################
    # Properties
    ####################################################################

    @property
    def density(self):
        return self._density

    @property
    def polynomial(self):
        return self._polynomial

    @property
    def dimension(self):
        return self._density.dimension

    ####################################################################
    # Core API
    ####################################################################

    def apply(self):

        if self._result is not None:
            return self._result

        spectral = self.spectral_decomposition()

        transformed = np.array(
            [
                self._polynomial(x)
                for x in spectral.eigenvalues
            ]
        )

        self._result = self.reconstruct(
            transformed
        )

        return self._result

    ####################################################################
    # Verification
    ####################################################################

    def transformed_eigenvalues(self):

        spectral = self.spectral_decomposition()

        return np.array(
            [
                self._polynomial(x)
                for x in spectral.eigenvalues
            ]
        )

    def verify_hermitian(self):

        A = self.apply().numpy()

        return np.allclose(
            A,
            A.conj().T,
            atol=1e-12,
        )

    ####################################################################
    # Export
    ####################################################################

    def numpy(self):
        return self.apply().numpy()

    ####################################################################
    # Representation
    ####################################################################

    def __repr__(self):

        poly_name = (
            "Polynomial"
            if isinstance(
                self._polynomial,
                Polynomial,
            )
            else "Callable"
        )

        return (
            "PolynomialEigenvalueTransformation("
            f"dimension={self.dimension}, "
            f"type={poly_name})"
        )