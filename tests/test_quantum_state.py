import numpy as np

from quantum_entropy.core.quantum_state import QuantumState


def test_basis_state():

    psi = QuantumState(
        np.array([1,0],dtype=complex)
    )

    assert psi.dimension == 2

    assert psi.num_qubits == 1

    assert psi.is_normalized()


def test_density_operator():

    psi = QuantumState(
        np.array([1,0],dtype=complex)
    )

    rho = psi.density_operator()

    assert rho.is_valid()


def test_tensor():

    a = QuantumState(
        np.array([1,0],dtype=complex)
    )

    b = QuantumState(
        np.array([0,1],dtype=complex)
    )

    c = a.tensor(b)

    assert c.dimension == 4


def test_fidelity():

    psi = QuantumState(
        np.array([1,0],dtype=complex)
    )

    phi = QuantumState(
        np.array([1,0],dtype=complex)
    )

    assert np.isclose(
        psi.fidelity(phi),
        1.0
    )


def test_probabilities():

    psi = QuantumState(
        np.array(
            [1/np.sqrt(2),1/np.sqrt(2)],
            dtype=complex
        )
    )

    probs = psi.probabilities()

    assert np.allclose(
        probs,
        [0.5,0.5]
    )