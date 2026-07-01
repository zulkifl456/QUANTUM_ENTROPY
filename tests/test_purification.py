import numpy as np

from quantum_entropy.core.density_operator import DensityOperator
from quantum_entropy.core.canonical_purification import CanonicalPurification
from quantum_entropy.core.quantum_state import QuantumState


def sample_density():

    rho = np.array(
        [
            [0.7, 0.2],
            [0.2, 0.3]
        ],
        dtype=complex
    )

    rho /= np.trace(rho)

    return DensityOperator(rho)


def test_constructor():

    rho = sample_density()

    purification = CanonicalPurification(rho)

    assert purification.dimension == 2

    assert purification.purified_dimension == 4


def test_state():

    rho = sample_density()

    purification = CanonicalPurification(rho)

    state = purification.state()

    assert isinstance(
        state,
        QuantumState
    )

    assert state.dimension == 4


def test_normalization():

    rho = sample_density()

    purification = CanonicalPurification(rho)

    state = purification.state()

    assert state.is_normalized()


def test_partial_trace():

    rho = sample_density()

    purification = CanonicalPurification(rho)

    recovered = purification.partial_trace()

    assert np.allclose(
        recovered.matrix,
        rho.matrix,
        atol=1e-12
    )


def test_verify():

    rho = sample_density()

    purification = CanonicalPurification(rho)

    assert purification.verify()