"""
==========================================================================
Quantum Entropy Algorithms Library

Module
------
Matrix Function

Description
-----------
Abstract base class for matrix-valued functions.

A matrix function acts spectrally,

    A = V Λ V†

and computes

    f(A) = V f(Λ) V†

Derived classes include

    • PolynomialEigenvalueTransformation
    • PositivePower
    • MatrixLogarithm
    • MatrixExponential

Author
------
Zulkifl Khairoowala
==========================================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from quantum_entropy.core.density_operator import DensityOperator


class MatrixFunction(ABC):
    """
    Base class for spectral matrix functions.
    """

    ####################################################################
    # Constructor
    ####################################################################

    def __init__(self, density: DensityOperator):

        if not isinstance(density, DensityOperator):
            raise TypeError(
                "Expected DensityOperator."
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
    def dimension(self):

        return self._density.dimension

    ####################################################################
    # Internal Helpers
    ####################################################################

    def spectral_decomposition(self):

        """
        Cached spectral decomposition.
        """

        return self._density.spectral_decomposition()

    def reconstruct(self, transformed_eigenvalues):

        """
        Reconstruct

            V diag(f(λ)) V†
        """

        spectral = self.spectral_decomposition()

        V = spectral.eigenvectors

        matrix = (
            V
            @ np.diag(transformed_eigenvalues)
            @ V.conj().T
        )

        return DensityOperator(matrix)

    ####################################################################
    # Interface
    ####################################################################

    @abstractmethod
    def apply(self):
        """
        Compute the matrix function.
        """

        raise NotImplementedError

    ####################################################################
    # Export
    ####################################################################

    def numpy(self):

        return self.apply().numpy()

    ####################################################################
    # Representation
    ####################################################################

    def __repr__(self):

        return (
            f"{self.__class__.__name__}"
            f"(dimension={self.dimension})"
        )