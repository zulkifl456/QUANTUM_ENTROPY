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

from quantum_entropy.primitives.polynomial import Polynomial

import numpy as np

from quantum_entropy.core.density_operator import DensityOperator


class PolynomialEigenvalueTransformation:
    """
    Numerical Polynomial Eigenvalue Transformation.
    """

    ####################################################################
    # Constructor
    ####################################################################

    def __init__(
        self,
        density: DensityOperator,
        polynomial,
    ):

        if not isinstance(density, DensityOperator):
            raise TypeError(
                "density must be a DensityOperator."
            )

        #
        # Accept either Polynomial or callable
        #
        if isinstance(polynomial, Polynomial):
            self._polynomial = polynomial

        elif callable(polynomial):
            self._polynomial = polynomial

        else:
            raise TypeError(
                "Expected Polynomial or callable."
            )

        self._density = density

        self._result = None

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
        """
        Apply the polynomial to the eigenvalues while
        preserving the eigenvectors.

        Returns
        -------
        DensityOperator
        """

        if self._result is not None:
            return self._result

        #
        # Spectral decomposition
        #
        spectral = self._density.spectral_decomposition()

        eigenvalues = spectral.eigenvalues
        eigenvectors = spectral.eigenvectors

        #
        # Apply polynomial
        #
        transformed = np.array(
            [
                self._polynomial(float(lam))
                for lam in eigenvalues
            ],
            dtype=np.float64,
        )

        #
        # Reconstruct matrix
        #
        matrix = (
            eigenvectors
            @ np.diag(transformed)
            @ eigenvectors.conj().T
        )

        self._result = DensityOperator(matrix)

        return self._result

    ####################################################################
    # Verification
    ####################################################################

    def transformed_eigenvalues(self):
        """
        Return P(λᵢ).
        """

        spectral = self._density.spectral_decomposition()

        return np.array(
            [
                self._polynomial(float(l))
                for l in spectral.eigenvalues
            ]
        )

    def verify_hermitian(self):
        """
        Verify that the transformed operator is Hermitian.
        """

        matrix = self.apply().numpy()

        return np.allclose(
            matrix,
            matrix.conj().T,
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