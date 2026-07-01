import numpy as np

from quantum_entropy.core.density_operator import DensityOperator
from quantum_entropy.primitives.matrix_function import MatrixFunction


class DummyMatrixFunction(MatrixFunction):

    def apply(self):

        return self.density


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


def test_constructor():

    f = DummyMatrixFunction(
        sample_density()
    )

    assert f.dimension == 2


def test_reconstruct():

    f = DummyMatrixFunction(
        sample_density()
    )

    spectral = (
        sample_density()
        .spectral_decomposition()
    )

    reconstructed = f.reconstruct(
        spectral.eigenvalues
    )

    assert np.allclose(
        reconstructed.numpy(),
        sample_density().numpy(),
        atol=1e-12,
    )


def test_numpy():

    f = DummyMatrixFunction(
        sample_density()
    )

    assert np.allclose(
        f.numpy(),
        sample_density().numpy(),
    )