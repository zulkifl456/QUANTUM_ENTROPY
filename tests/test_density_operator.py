import numpy as np

from quantum_entropy.core.density_operator import DensityOperator


def test_identity_density():

    rho = np.eye(2) / 2

    state = DensityOperator(rho)

    assert state.dimension == 2

    assert state.shape == (2, 2)

    assert state.is_hermitian()

    assert state.has_unit_trace()

    assert state.is_positive()

    assert state.is_valid()

    assert np.isclose(
        state.trace(),
        1.0
    )