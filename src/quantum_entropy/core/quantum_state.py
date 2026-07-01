"""
==========================================================================
Quantum Entropy Algorithms Library

Module
------
Quantum State

Description
-----------
Represents a pure quantum state.

A QuantumState is the pure-state counterpart of DensityOperator.

This class will be used throughout the library by

    • Canonical Purification
    • Purified Query Oracle
    • Block Encoding
    • Reflection Operators
    • Signal Operators
    • QSVT
    • Amplitude Estimation

Author
------
Zulkifl Khairoowala

==========================================================================
"""

from __future__ import annotations

import numpy as np

from quantum_entropy.core.density_operator import DensityOperator


class QuantumState:

    TOL = 1e-12

    ####################################################################
    # Constructor
    ####################################################################

    def __init__(self, statevector):

        self._state = np.asarray(
            statevector,
            dtype=np.complex128
        ).flatten()

        if self._state.ndim != 1:
            raise ValueError(
                "Statevector must be one-dimensional."
            )

        if self._state.size == 0:
            raise ValueError(
                "Statevector cannot be empty."
            )

        if not self.is_normalized():
            raise ValueError(
                "Statevector must be normalized."
            )

    ####################################################################
    # Properties
    ####################################################################

    @property
    def statevector(self):

        return self._state.copy()

    @property
    def dimension(self):

        return self._state.size

    @property
    def num_qubits(self):

        n = np.log2(self.dimension)

        if not np.isclose(n, round(n)):
            raise ValueError(
                "Dimension is not a power of two."
            )

        return int(round(n))

    ####################################################################
    # Validation
    ####################################################################

    def norm(self):

        return float(
            np.linalg.norm(self._state)
        )

    def is_normalized(self):

        return np.isclose(
            self.norm(),
            1.0,
            atol=self.TOL
        )

    ####################################################################
    # Linear Algebra
    ####################################################################

    def inner(self, other):

        if not isinstance(other, QuantumState):
            raise TypeError(
                "Expected QuantumState."
            )

        return np.vdot(
            self._state,
            other._state
        )

    def tensor(self, other):

        if not isinstance(other, QuantumState):
            raise TypeError(
                "Expected QuantumState."
            )

        return QuantumState(
            np.kron(
                self._state,
                other._state
            )
        )

    ####################################################################
    # Density Matrix
    ####################################################################

    def density_operator(self):

        rho = np.outer(
            self._state,
            self._state.conj()
        )

        return DensityOperator(rho)

    ####################################################################
    # Measurement
    ####################################################################

    def probabilities(self):

        return np.abs(
            self._state
        ) ** 2

    ####################################################################
    # Bloch Vector
    ####################################################################

    def bloch_vector(self):

        if self.dimension != 2:
            raise ValueError(
                "Bloch vector only defined for one qubit."
            )

        rho = self.density_operator().matrix

        X = np.array([[0,1],[1,0]],dtype=complex)
        Y = np.array([[0,-1j],[1j,0]],dtype=complex)
        Z = np.array([[1,0],[0,-1]],dtype=complex)

        return np.array([

            np.real(np.trace(rho @ X)),

            np.real(np.trace(rho @ Y)),

            np.real(np.trace(rho @ Z))

        ])

    ####################################################################
    # Fidelity
    ####################################################################

    def fidelity(self, other):

        return abs(
            self.inner(other)
        ) ** 2

    ####################################################################
    # Export
    ####################################################################

    def numpy(self):

        return self.statevector

    def copy(self):

        return QuantumState(
            self.statevector
        )

    ####################################################################
    # Representation
    ####################################################################

    def __repr__(self):

        return (

            "QuantumState("

            f"dimension={self.dimension}, "

            f"qubits={self.num_qubits}"

            ")"
        )