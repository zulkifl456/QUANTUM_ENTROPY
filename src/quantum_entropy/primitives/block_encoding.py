"""
==========================================================================
Quantum Entropy Algorithms Library

Module
------
Block Encoding

Description
-----------
Represents a block encoding of a density operator.

Given a state preparation oracle Uρ,

the block encoding satisfies

    (<0| ⊗ I) U_B (|0> ⊗ I) = ρ

This class currently defines only the public API.

Reference
---------
New Quantum Algorithms for Computing Quantum Entropies
and Distances of Density Operators

==========================================================================
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from quantum_entropy.core.state_preparation_oracle import (
    StatePreparationOracle,
)


class BlockEncoding:
    """
    Block encoding of a density operator.
    """

    ####################################################################
    # Constructor
    ####################################################################

    def __init__(self, oracle: StatePreparationOracle):

        if not isinstance(
            oracle,
            StatePreparationOracle,
        ):
            raise TypeError(
                "Expected StatePreparationOracle."
            )

        self._oracle = oracle

        self._block_matrix: Optional[np.ndarray] = None

    ####################################################################
    # Properties
    ####################################################################

    @property
    def oracle(self):

        return self._oracle

    @property
    def dimension(self):

        return self._oracle.dimension

    @property
    def num_qubits(self):

        return self._oracle.num_qubits

    ####################################################################
    # Core
    ####################################################################

    def matrix(self):
        """
        Return the block encoding unitary.
        """

        raise NotImplementedError

    ####################################################################
    # Derived Objects
    ####################################################################

    def top_left_block(self):
        """
        Return the encoded block.

        Should equal the density operator.
        """

        raise NotImplementedError

    ####################################################################
    # Verification
    ####################################################################

    def verify_unitary(self):
        """
        Verify U†U = I.
        """

        raise NotImplementedError

    def verify_encoding(self):
        """
        Verify the top-left block equals the
        encoded density operator.
        """

        raise NotImplementedError

    ####################################################################
    # Export
    ####################################################################

    def numpy(self):

        return self.matrix()

    ####################################################################
    # Representation
    ####################################################################

    def __repr__(self):

        return (

            "BlockEncoding("

            f"dimension={self.dimension}, "

            f"qubits={self.num_qubits}"

            ")"

        )