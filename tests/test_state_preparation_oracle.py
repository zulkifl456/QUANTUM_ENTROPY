import numpy as np

from quantum_entropy.core.quantum_state import QuantumState
from quantum_entropy.core.state_preparation_oracle import (
    StatePreparationOracle
)


def sample_state():

    psi = np.array(
        [0.5, 0.5, 0.5, 0.5],
        dtype=complex
    )

    psi /= np.linalg.norm(psi)

    return QuantumState(psi)


def test_constructor():

    oracle = StatePreparationOracle(
        sample_state()
    )

    assert oracle.dimension == 4


def test_unitary():

    oracle = StatePreparationOracle(
        sample_state()
    )

    U = oracle.unitary()

    I = np.eye(U.shape[0])

    assert np.allclose(
        U.conj().T @ U,
        I,
        atol=1e-12
    )


def test_prepare():

    oracle = StatePreparationOracle(
        sample_state()
    )

    prepared = oracle.prepare()

    assert np.allclose(
        prepared.numpy(),
        sample_state().numpy(),
        atol=1e-12
    )


def test_inverse():

    oracle = StatePreparationOracle(
        sample_state()
    )

    U = oracle.unitary()

    Uinv = oracle.inverse()

    assert np.allclose(
        U @ Uinv,
        np.eye(U.shape[0]),
        atol=1e-12
    )


def test_controlled():

    oracle = StatePreparationOracle(
        sample_state()
    )

    CU = oracle.controlled()

    assert CU.shape == (8, 8)


def test_verify():

    oracle = StatePreparationOracle(
        sample_state()
    )

    assert oracle.verify()