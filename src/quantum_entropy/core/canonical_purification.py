"""
==========================================================================
Quantum Entropy Algorithms Library

Module
------
Canonical Purification

Description
-----------
Defines the canonical purification of a quantum density operator.

The canonical purification is

    |ρ> = Σ_i √λ_i |v_i> ⊗ |v_i>

where

    λ_i : eigenvalues
    |v_i> : eigenvectors

This class currently defines only the public API.
The numerical implementation will be added in Commit 2.

Reference
---------
New Quantum Algorithms for Computing Quantum Entropies
and Distances of Density Operators

Author
------
Zulkifl Khairoowala

==========================================================================
"""

from __future__ import annotations

import numpy as np

from quantum_entropy.core.density_operator import DensityOperator


class CanonicalPurification:
    """
    Canonical purification of a density operator.

    Parameters
    ----------
    density_operator : DensityOperator

        Density operator to purify.
    """

    ####################################################################
    # Constructor
    ####################################################################

    def __init__(self, density_operator: DensityOperator):

        if not isinstance(
            density_operator,
            DensityOperator
        ):
            raise TypeError(
                "Input must be a DensityOperator."
            )

        self._density = density_operator

        self._statevector = None

    ####################################################################
    # Properties
    ####################################################################

    @property
    def density_operator(self):

        """Return original density operator."""

        return self._density

    @property
    def dimension(self):

        """Dimension of original Hilbert space."""

        return self._density.dimension

    @property
    def purified_dimension(self):

        """Dimension of purified Hilbert space."""

        return self.dimension ** 2

    ####################################################################
    # Core API
    ####################################################################

    def statevector(self):

        """
        Return canonical purification statevector.

        Returns
        -------
        ndarray
        """

        raise NotImplementedError

    def density_matrix(self):

        """
        Density matrix of purified state.

        Returns
        -------
        ndarray
        """

        raise NotImplementedError

    ####################################################################
    # Verification
    ####################################################################

    def partial_trace(self):

        """
        Recover original density matrix
        by tracing out the ancilla.

        Returns
        -------
        ndarray
        """

        raise NotImplementedError

    def verify(self):

        """
        Verify that

        Tr_anc(|ρ><ρ|)=ρ

        Returns
        -------
        bool
        """

        raise NotImplementedError

    ####################################################################
    # Export
    ####################################################################

    def numpy(self):

        """
        Export purified statevector.

        Returns
        -------
        ndarray
        """

        raise NotImplementedError

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