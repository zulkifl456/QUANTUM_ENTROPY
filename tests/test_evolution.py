import numpy as np

from quantum_entropy.core.density_operator import DensityOperator
from quantum_entropy.core.quantum_state import QuantumState
from quantum_entropy.core.state_preparation_oracle import (
    StatePreparationOracle,
)
from quantum_entropy.primitives.block_encoding import (
    BlockEncoding,
)
from quantum_entropy.primitives.evolution import (
    Evolution,
)


class DummyBlockEncoding(BlockEncoding):

    def matrix(self):
        return np.eye(4)

    def top_left_block(self):
        return np.eye(4)

    def verify_unitary(self):
        return True

    def verify_encoding(self):
        return True


def sample_density():

    rho = np.diag(
        [0.4, 0.3, 0.2, 0.1]
    )

    return DensityOperator(rho)


def sample_encoding():

    psi = QuantumState(
        np.array(
            [1, 0, 0, 0],
            dtype=complex,
        )
    )

    oracle = StatePreparationOracle(psi)

    return DummyBlockEncoding(oracle)


def test_constructor():

    evo = Evolution(
        sample_density(),
        sample_encoding(),
    )

    assert evo.density.dimension == 4


def test_identity_evolution():

    evo = Evolution(
        sample_density(),
        sample_encoding(),
    )

    evolved = evo.evolve()

    assert np.allclose(
        evolved.numpy(),
        sample_density().numpy(),
    )


def test_hermitian():

    evo = Evolution(
        sample_density(),
        sample_encoding(),
    )

    assert evo.verify_hermitian()


def test_positive():

    evo = Evolution(
        sample_density(),
        sample_encoding(),
    )

    assert evo.verify_positive()


def test_trace():

    evo = Evolution(
        sample_density(),
        sample_encoding(),
    )

    assert np.isclose(
        evo.trace(),
        1.0,
    )