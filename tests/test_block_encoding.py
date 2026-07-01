import numpy as np

from quantum_entropy.core.quantum_state import QuantumState
from quantum_entropy.core.state_preparation_oracle import (
    StatePreparationOracle,
)
from quantum_entropy.primitives.block_encoding import (
    BlockEncoding,
)


def sample_oracle():

    psi = np.array(
        [0.5, 0.5, 0.5, 0.5],
        dtype=complex,
    )

    psi /= np.linalg.norm(psi)

    state = QuantumState(psi)

    return StatePreparationOracle(state)


def test_constructor():

    oracle = sample_oracle()

    block = BlockEncoding(oracle)

    assert block.dimension == 4

    assert block.num_qubits == 2

    assert block.oracle is oracle