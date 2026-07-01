"""
==========================================================================
Quantum Entropy Algorithms Library

Module
------
State Preparation Oracle

Description
-----------
Represents the unitary

    U |0...0> = |ψ>

where |ψ> is typically the canonical purification of a density operator.

This oracle is the first quantum object required by the entropy
algorithms and serves as the foundation for

    • Block Encoding
    • QSVT
    • Amplitude Estimation

Author
------
Zulkifl Khairoowala
==========================================================================
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from quantum_entropy.core.quantum_state import QuantumState


class StatePreparationOracle:
    """
    State preparation oracle.

    Parameters
    ----------
    state : QuantumState
        Target quantum state.
    """

    ####################################################################
    # Constructor
    ####################################################################

    def __init__(self, state: QuantumState):

        if not isinstance(state, QuantumState):
            raise TypeError("Expected QuantumState.")

        self._state = state

        #
        # Lazy cache
        #
        self._unitary: Optional[np.ndarray] = None

    ####################################################################
    # Properties
    ####################################################################

    @property
    def state(self):
        return self._state

    @property
    def dimension(self):
        return self._state.dimension

    @property
    def num_qubits(self):
        return self._state.num_qubits

    ####################################################################
    # Core API
    ####################################################################

    def unitary(self):
        """
        Construct a unitary U such that

            U|0...0> = |ψ|

        using a Householder reflection.

        Returns
        -------
        ndarray
        """

        if self._unitary is not None:
            return self._unitary

        psi = self._state.numpy()

        n = psi.size

        e0 = np.zeros(n, dtype=np.complex128)
        e0[0] = 1.0

        #
        # If already |0>, oracle is identity.
        #
        if np.allclose(psi, e0):

            self._unitary = np.eye(
                n,
                dtype=np.complex128,
            )

            return self._unitary

        #
        # Householder vector
        #
        v = e0 - psi

        norm = np.linalg.norm(v)

        if norm < 1e-14:

            self._unitary = np.eye(
                n,
                dtype=np.complex128,
            )

            return self._unitary

        v /= norm

        #
        # Householder reflection
        #
        H = np.eye(
            n,
            dtype=np.complex128,
        ) - 2 * np.outer(v, v.conj())

        self._unitary = H

        return self._unitary

    ####################################################################
    # Derived Operations
    ####################################################################

    def inverse(self):
        """
        Return U†.
        """

        return self.unitary().conj().T

    def controlled(self):
        """
        Controlled state-preparation oracle.

        Returns
        -------
        ndarray
        """

        U = self.unitary()

        n = U.shape[0]

        CU = np.zeros(
            (2 * n, 2 * n),
            dtype=np.complex128,
        )

        CU[:n, :n] = np.eye(
            n,
            dtype=np.complex128,
        )

        CU[n:, n:] = U

        return CU

    def prepare(self):
        """
        Apply U to |0...0>.

        Returns
        -------
        QuantumState
        """

        U = self.unitary()

        zero = np.zeros(
            self.dimension,
            dtype=np.complex128,
        )

        zero[0] = 1.0

        psi = U @ zero

        return QuantumState(psi)

    ####################################################################
    # Verification
    ####################################################################

    def verify(self):
        """
        Verify

            U|0...0> = |ψ|

        Returns
        -------
        bool
        """

        prepared = self.prepare()

        return np.allclose(
            prepared.numpy(),
            self._state.numpy(),
            atol=1e-12,
        )

    ####################################################################
    # Export
    ####################################################################

    def numpy(self):
        return self.unitary()

    ####################################################################
    # Representation
    ####################################################################

    def __repr__(self):

        return (
            "StatePreparationOracle("
            f"dimension={self.dimension}, "
            f"qubits={self.num_qubits}"
            ")"
        )