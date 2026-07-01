import numpy as np

from quantum_entropy.core.density_operator import DensityOperator
from quantum_entropy.core.canonical_purification import (
    CanonicalPurification,
)
from quantum_entropy.core.state_preparation_oracle import (
    StatePreparationOracle,
)
from quantum_entropy.primitives.block_encoding_circuit import (
    BlockEncodingCircuit,
)


def sample_density():

    rho = np.array(
        [
            [0.7, 0.2],
            [0.2, 0.3],
        ],
        dtype=complex,
    )

    rho /= np.trace(rho)

    return DensityOperator(rho)


def sample_oracle():

    purification = CanonicalPurification(
        sample_density()
    )

    return StatePreparationOracle(
        purification.state()      # <-- IMPORTANT
    )


def test_constructor():

    oracle = sample_oracle()

    circuit = BlockEncodingCircuit(
        oracle
    )

    assert circuit.oracle is oracle


def test_system_qubits():

    oracle = sample_oracle()

    circuit = BlockEncodingCircuit(
        oracle
    )

    assert circuit.system_qubits == 2


def test_total_qubits():

    oracle = sample_oracle()

    circuit = BlockEncodingCircuit(
        oracle
    )

    assert circuit.total_qubits == 4


def test_repr():

    oracle = sample_oracle()

    circuit = BlockEncodingCircuit(
        oracle
    )

    assert "BlockEncodingCircuit" in repr(circuit)