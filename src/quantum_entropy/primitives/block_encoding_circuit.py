"""
==========================================================================
Quantum Entropy Algorithms Library

Module
------
Block Encoding Circuit

Description
-----------
Represents the quantum circuit described in Lemma II.1.

Given a purification oracle U preparing a subnormalized
density operator, this class stores the quantum resources
required to construct the corresponding block encoding.

The actual unitary construction is implemented in the
next commit.

Author
------
Zulkifl Khairoowala
==========================================================================
"""

from __future__ import annotations

from quantum_entropy.core.state_preparation_oracle import (
    StatePreparationOracle,
)


class BlockEncodingCircuit:
    """
    Quantum circuit implementing Lemma II.1.
    """

    def __init__(self, oracle: StatePreparationOracle):

        if not isinstance(
            oracle,
            StatePreparationOracle,
        ):
            raise TypeError(
                "Expected StatePreparationOracle."
            )

        self._oracle = oracle

    @property
    def oracle(self):

        return self._oracle

    @property
    def system_qubits(self):

        return self.oracle.num_qubits

    @property
    def ancilla_qubits(self):

        #
        # Lemma II.1 eventually uses
        # n + a ancillas.
        #
        return self.oracle.num_qubits

    @property
    def total_qubits(self):

        return (
            self.system_qubits
            + self.ancilla_qubits
        )

    def __repr__(self):

        return (
            "BlockEncodingCircuit("
            f"system={self.system_qubits}, "
            f"ancilla={self.ancilla_qubits})"
        )