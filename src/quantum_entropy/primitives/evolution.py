"""
===========================================================================
Quantum Entropy Algorithms Library

Lemma II.2
Evolution of Subnormalized Density Operators

Reference
---------
Q. Wang et al.
New Quantum Algorithms for Computing Quantum Entropies
and Distances of Density Operators

IEEE Transactions on Information Theory, 2024

This module implements the numerical analogue of

    A  --->  B A B†

where B is represented through a block encoding.

===========================================================================
"""

from __future__ import annotations

import numpy as np

from quantum_entropy.core.density_operator import DensityOperator
from quantum_entropy.primitives.block_encoding import BlockEncoding


class Evolution:
    """
    Numerical implementation of Lemma II.2.
    """

    def __init__(
        self,
        density: DensityOperator,
        block_encoding: BlockEncoding,
    ):

        if not isinstance(density, DensityOperator):
            raise TypeError(
                "density must be a DensityOperator."
            )

        if not isinstance(block_encoding, BlockEncoding):
            raise TypeError(
                "block_encoding must be a BlockEncoding."
            )

        if density.dimension != block_encoding.dimension:
            raise ValueError(
                "Dimension mismatch."
            )

        self._density = density
        self._encoding = block_encoding

    @property
    def density(self):

        return self._density

    @property
    def block_encoding(self):

        return self._encoding

    def evolve(self):
        """
        Compute

            B A B†

        where B is the encoded operator.
        """

        A = self._density.numpy()

        B = self._encoding.top_left_block()

        evolved = B @ A @ B.conj().T

        return DensityOperator(evolved)

    def verify_hermitian(self):

        evolved = self.evolve().numpy()

        return np.allclose(
            evolved,
            evolved.conj().T,
            atol=1e-12,
        )

    def verify_positive(self):

        eigvals = np.linalg.eigvalsh(
            self.evolve().numpy()
        )

        return np.all(eigvals >= -1e-12)

    def trace(self):

        return np.trace(
            self.evolve().numpy()
        ).real

    def __repr__(self):

        return (
            f"Evolution("
            f"dimension={self.density.dimension})"
        )