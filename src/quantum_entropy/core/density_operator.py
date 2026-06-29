"""
==========================================================================
Quantum Entropy Algorithms Library

Module
------
Density Operator

Description
-----------
Core implementation of quantum density operators.

This class performs

    • Validation
    • Spectral decomposition
    • Cached eigendecomposition
    • Matrix functions

Author
------
Zulkifl Khairoowala
==========================================================================
"""

from __future__ import annotations

import numpy as np

from quantum_entropy.core.spectral import SpectralDecomposition


class DensityOperator:

    TOL = 1e-12

    ####################################################################
    # Constructor
    ####################################################################

    def __init__(self, matrix):

        self._matrix = np.asarray(
            matrix,
            dtype=np.complex128
        )

        if self._matrix.ndim != 2:
            raise ValueError(
                "Density matrix must be two-dimensional."
            )

        if self._matrix.shape[0] != self._matrix.shape[1]:
            raise ValueError(
                "Density matrix must be square."
            )

        self._spectral_cache = None

    ####################################################################
    # Properties
    ####################################################################

    @property
    def matrix(self):

        return self._matrix.copy()

    @property
    def dimension(self):

        return self._matrix.shape[0]

    @property
    def shape(self):

        return self._matrix.shape

    ####################################################################
    # Validation
    ####################################################################

    def is_hermitian(self):

        return np.allclose(
            self._matrix,
            self._matrix.conj().T,
            atol=self.TOL
        )

    def has_unit_trace(self):

        return np.isclose(
            np.trace(self._matrix),
            1.0,
            atol=self.TOL
        )

    def is_positive(self):

        if not self.is_hermitian():
            return False

        eigvals = np.linalg.eigvalsh(
            self._matrix
        )

        return np.all(
            eigvals >= -self.TOL
        )

    def is_valid(self):

        return (

            self.is_hermitian()

            and

            self.has_unit_trace()

            and

            self.is_positive()

        )

    ####################################################################
    # Basic Linear Algebra
    ####################################################################

    def trace(self):

        return float(
            np.trace(self._matrix).real
        )

    ####################################################################
    # Spectral Engine
    ####################################################################

    def spectral_decomposition(self):

        """
        Compute (or retrieve cached)
        spectral decomposition.

        Returns
        -------
        SpectralDecomposition
        """

        if self._spectral_cache is None:

            eigvals, eigvecs = np.linalg.eigh(
                self._matrix
            )

            eigvals = np.real_if_close(
                eigvals
            )

            self._spectral_cache = SpectralDecomposition(

                eigenvalues=eigvals,

                eigenvectors=eigvecs

            )

        return self._spectral_cache

    ####################################################################
    # Spectral Quantities
    ####################################################################

    def eigenvalues(self):

        return self.spectral_decomposition().eigenvalues.copy()

    def eigenvectors(self):

        return self.spectral_decomposition().eigenvectors.copy()

    def rank(self):

        return self.spectral_decomposition().rank

    ####################################################################
    # Matrix Functions
    ####################################################################

    def sqrt(self):

        return self.spectral_decomposition().sqrt()

    def inverse(self):

        return self.spectral_decomposition().inverse()

    def power(self, alpha):

        return self.spectral_decomposition().power(alpha)

    def log(self, base=2):

        return self.spectral_decomposition().log(base)

    ####################################################################
    # Future Modules
    ####################################################################

    def purity(self):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def linear_entropy(self):
        raise NotImplementedError

    ####################################################################
    # Export
    ####################################################################

    def copy(self):

        return DensityOperator(
            self.matrix
        )

    def numpy(self):

        return self.matrix

    ####################################################################
    # Representation
    ####################################################################

    def __repr__(self):

        return (

            "DensityOperator("

            f"dimension={self.dimension}, "

            f"rank={self.rank()})"

        )