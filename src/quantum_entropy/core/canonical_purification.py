"""
==========================================================================
Quantum Entropy Algorithms Library

Module
------
Canonical Purification

Description
-----------
Canonical purification

    |ρ> = Σ_i √λ_i |v_i> ⊗ |v_i>

This class owns the purified quantum state.

Author
------
Zulkifl Khairoowala
==========================================================================
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from quantum_entropy.core.density_operator import DensityOperator
from quantum_entropy.core.quantum_state import QuantumState


class CanonicalPurification:
    """
    Canonical purification of a density operator.

    Given the spectral decomposition

        ρ = Σ_i λ_i |v_i><v_i|,

    the canonical purification is

        |ρ> = Σ_i √λ_i |v_i> ⊗ |v_i|.
    """

    ####################################################################
    # Constructor
    ####################################################################

    def __init__(self, density_operator: DensityOperator):

        if not isinstance(density_operator, DensityOperator):
            raise TypeError("Expected DensityOperator.")

        self._density = density_operator

        #
        # Lazy cache
        #
        self._state: Optional[QuantumState] = None

    ####################################################################
    # Properties
    ####################################################################

    @property
    def density_operator(self):
        return self._density

    @property
    def dimension(self):
        return self._density.dimension

    @property
    def purified_dimension(self):
        return self.dimension * self.dimension

    ####################################################################
    # Core
    ####################################################################

    def state(self) -> QuantumState:
        """
        Construct (or return cached) canonical purified state.
        """

        #
        # Lazy cache
        #
        if self._state is not None:
            return self._state

        #
        # Spectral decomposition
        #
        spectral = self._density.spectral_decomposition()

        eigenvalues = spectral.eigenvalues
        eigenvectors = spectral.eigenvectors

        dimension = self.dimension

        #
        # Purified state
        #
        psi = np.zeros(
            dimension * dimension,
            dtype=np.complex128,
        )

        for i, lam in enumerate(eigenvalues):

            if lam <= 0:
                continue

            vi = eigenvectors[:, i]

            psi += np.sqrt(lam) * np.kron(vi, vi)

        #
        # Numerical normalization
        #
        psi /= np.linalg.norm(psi)

        self._state = QuantumState(psi)

        return self._state

    ####################################################################
    # Derived Objects
    ####################################################################

    def statevector(self):
        """
        Convenience wrapper.

        Returns
        -------
        ndarray
        """

        return self.state().numpy()

    def density_matrix(self):
        """
        Return |ρ><ρ|.
        """

        return self.state().density_operator()

    ####################################################################
    # Verification
    ####################################################################

    def partial_trace(self):
        """
        Compute Tr₂(|ρ><ρ|).

        For the canonical purification this equals the original
        density operator.
        """

        psi = self.state().numpy()

        d = self.dimension

        psi = psi.reshape(d, d)

        rho = psi @ psi.conj().T

        return DensityOperator(rho)

    def verify(self):
        """
        Verify that

            Tr₂(|ρ><ρ|) = ρ.
        """

        recovered = self.partial_trace().matrix

        return np.allclose(
            recovered,
            self._density.matrix,
            atol=1e-12,
        )

    ####################################################################
    # Export
    ####################################################################

    def numpy(self):
        return self.statevector()

    ####################################################################
    # Representation
    ####################################################################

    def __repr__(self):

        return (
            "CanonicalPurification("
            f"dimension={self.dimension}, "
            f"purified_dimension={self.purified_dimension}"
            ")"
        )