"""
==========================================================================
Quantum Entropy Algorithms Library

Module:
    Density Operator

Commit:
    2 - Validation Layer

Author:
    Zulkifl Khairoowala
==========================================================================
"""

from __future__ import annotations

import numpy as np


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

        eigvals = np.linalg.eigvalsh(self._matrix)

        return np.all(eigvals >= -self.TOL)

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

        return np.trace(self._matrix).real

    ####################################################################
    # Placeholders
    ####################################################################

    def rank(self):
        raise NotImplementedError

    def eigenvalues(self):
        raise NotImplementedError

    def eigenvectors(self):
        raise NotImplementedError

    def spectral_decomposition(self):
        raise NotImplementedError

    def sqrt(self):
        raise NotImplementedError

    def log(self):
        raise NotImplementedError

    def inverse(self):
        raise NotImplementedError

    def power(self, alpha):
        raise NotImplementedError

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
            self._matrix.copy()
        )

    def numpy(self):

        return self.matrix

    ####################################################################
    # Representation
    ####################################################################

    def __repr__(self):

        status = "Valid" if self.is_valid() else "Invalid"

        return (

            f"DensityOperator("

            f"dimension={self.dimension}, "

            f"{status})"

        )