"""
==========================================================================
Quantum Entropy Algorithms Library

Module:
    Spectral Decomposition

Description
-----------
Defines the immutable spectral decomposition of a Hermitian matrix.

This module is shared by:

    • DensityOperator
    • Canonical Purification
    • Block Encoding
    • QSVT
    • Quantum Information Algorithms

Author
------
Zulkifl Khairoowala

==========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class SpectralDecomposition:
    """
    Immutable spectral decomposition of a Hermitian matrix.

    Parameters
    ----------
    eigenvalues : ndarray
        Eigenvalues sorted in ascending order.

    eigenvectors : ndarray
        Corresponding orthonormal eigenvectors stored column-wise.
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray

    ####################################################################
    # Basic Properties
    ####################################################################

    @property
    def dimension(self) -> int:
        """Dimension of the Hilbert space."""

        return self.eigenvalues.size

    @property
    def rank(self) -> int:
        """Numerical matrix rank."""

        eps = np.finfo(float).eps

        return int(np.sum(self.eigenvalues > eps))

    @property
    def trace(self) -> float:
        """Trace from eigenvalues."""

        return float(np.sum(self.eigenvalues))

    ####################################################################
    # Matrix Reconstruction
    ####################################################################

    def reconstruct(self) -> np.ndarray:
        """
        Reconstruct the original matrix.

        Returns
        -------
        ndarray
        """

        return (
            self.eigenvectors
            @ np.diag(self.eigenvalues)
            @ self.eigenvectors.conj().T
        )

    ####################################################################
    # Matrix Functions
    ####################################################################

    def power(self, alpha: float) -> np.ndarray:
        """
        Compute matrix power.

        Returns

            V diag(lambda^alpha) V†
        """

        vals = np.power(self.eigenvalues, alpha)

        return (
            self.eigenvectors
            @ np.diag(vals)
            @ self.eigenvectors.conj().T
        )

    def sqrt(self) -> np.ndarray:
        """Matrix square root."""

        return self.power(0.5)

    def inverse(self) -> np.ndarray:
        """
        Matrix inverse.

        Raises
        ------
        LinAlgError
            If matrix is singular.
        """

        if np.any(np.isclose(self.eigenvalues, 0.0)):
            raise np.linalg.LinAlgError(
                "Matrix is singular."
            )

        vals = 1.0 / self.eigenvalues

        return (
            self.eigenvectors
            @ np.diag(vals)
            @ self.eigenvectors.conj().T
        )

    def log(self, base: float = 2.0) -> np.ndarray:
        """
        Matrix logarithm.

        Parameters
        ----------
        base : float
            Logarithm base.

        Returns
        -------
        ndarray
        """

        if np.any(self.eigenvalues <= 0):
            raise ValueError(
                "Matrix logarithm undefined for non-positive eigenvalues."
            )

        vals = np.log(self.eigenvalues) / np.log(base)

        return (
            self.eigenvectors
            @ np.diag(vals)
            @ self.eigenvectors.conj().T
        )

    ####################################################################
    # Validation
    ####################################################################

    def is_orthonormal(self, atol: float = 1e-12) -> bool:
        """
        Check whether eigenvectors are orthonormal.
        """

        identity = np.eye(self.dimension)

        return np.allclose(
            self.eigenvectors.conj().T @ self.eigenvectors,
            identity,
            atol=atol
        )

    ####################################################################
    # Representation
    ####################################################################

    def __repr__(self):

        return (
            "SpectralDecomposition("
            f"dimension={self.dimension}, "
            f"rank={self.rank})"
        )
    