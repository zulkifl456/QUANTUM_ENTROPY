from quantum_entropy.core.density_operator import DensityOperator
from quantum_entropy.core.canonical_purification import CanonicalPurification

import numpy as np


def test_constructor():

    rho = DensityOperator(
        np.eye(2) / 2
    )

    purification = CanonicalPurification(rho)

    assert purification.dimension == 2

    assert purification.purified_dimension == 4

    assert purification.density_operator is rho