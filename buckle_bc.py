from typing import Any, Optional, TypeVar, Generic

import numpy as np
from numpy.typing import NDArray

# from abc import ABC, abstractmethod

from numba import njit

# from elastica._linalg import _batch_matvec, _batch_matrix_transpose
# from elastica._rotations import _get_rotation_matrix
import elastica as ea
from elastica.typing import SystemType, RodType, RigidBodyType, ConstrainingIndex

class BuckleBC(ea.GeneralConstraint):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """

        Initialization of the constraint. Any parameter passed to 'using' will be available in kwargs.

        Parameters
        ----------
        constrained_position_idx : tuple
            Tuple of position-indices that will be constrained
        constrained_director_idx : tuple
            Tuple of director-indices that will be constrained
        """
        super().__init__(
            *args,
            translational_constraint_selector=np.array([True, True, True]),
            rotational_constraint_selector=np.array([False, False, False]),
            **kwargs,
        )

    def constrain_values(
        self, system: "RodType | RigidBodyType", time: np.float64
    ) -> None:
        if self.constrained_position_idx.size:
            self.nb_constrain_translational_values(
                system.position_collection,
                self.fixed_positions,
                self.constrained_position_idx,
            )
        if self.constrained_director_idx.size:
            self.nb_constraint_rotational_values(
                system.director_collection,
                self.fixed_directors,
                self.constrained_director_idx,
            )

    def constrain_rates(
        self, system: "RodType | RigidBodyType", time: np.float64
    ) -> None:
        if self.constrained_position_idx.size:
            self.nb_constrain_translational_rates(
                system.velocity_collection,
                self.constrained_position_idx,
            )
        if self.constrained_director_idx.size:
            self.nb_constrain_rotational_rates(
                system.omega_collection,
                self.constrained_director_idx,
            )

    @staticmethod
    @njit(cache=True)  # type: ignore
    def nb_constraint_rotational_values(
        director_collection: NDArray[np.float64],
        fixed_director_collection: NDArray[np.float64],
        indices: NDArray[np.int32],
    ) -> None:
        """
        Computes constrain values in numba njit decorator
        Parameters
        ----------
        director_collection : numpy.ndarray
            3D (dim, dim, blocksize) array containing data with `float` type.
        fixed_director_collection : numpy.ndarray
            3D (dim, dim, blocksize) array containing data with `float` type.
        indices : numpy.ndarray
            1D array containing the index of constraining nodes
        """
        block_size = indices.size
        for i in range(block_size):
            k = indices[i]
            director_collection[..., k] = fixed_director_collection[..., i]

    @staticmethod
    @njit(cache=True)  # type: ignore
    def nb_constrain_translational_values(
        position_collection: NDArray[np.float64],
        fixed_position_collection: NDArray[np.float64],
        indices: NDArray[np.int32],
    ) -> None:
        """
        Computes constrain values in numba njit decorator
        Parameters
        ----------
        position_collection : numpy.ndarray
            2D (dim, blocksize) array containing data with `float` type.
        fixed_position_collection : numpy.ndarray
            2D (dim, blocksize) array containing data with `float` type.
        indices : numpy.ndarray
            1D array containing the index of constraining nodes
        """

        position_collection[0, 0] = fixed_position_collection[0, 0]
        position_collection[1, 0] = fixed_position_collection[1, 0]
        position_collection[2, 0] = fixed_position_collection[2, 0]

        position_collection[0, -1] = fixed_position_collection[0, -1]
        position_collection[2, -1] = fixed_position_collection[2, -1]

    @staticmethod
    @njit(cache=True)  # type: ignore
    def nb_constrain_translational_rates(
        velocity_collection: NDArray[np.float64], indices: NDArray[np.int32]
    ) -> None:
        """
        Compute constrain rates in numba njit decorator
        Parameters
        ----------
        velocity_collection : numpy.ndarray
            2D (dim, blocksize) array containing data with `float` type.
        indices : numpy.ndarray
            1D array containing the index of constraining nodes
        """

        velocity_collection[0, 0] = 0.0
        velocity_collection[1, 0] = 0.0
        velocity_collection[2, 0] = 0.0
        
        velocity_collection[0, -1] = 0.0
        velocity_collection[2, -1] = 0.0

    @staticmethod
    @njit(cache=True)  # type: ignore
    def nb_constrain_rotational_rates(
        omega_collection: NDArray[np.float64], indices: NDArray[np.int32]
    ) -> None:
        """
        Compute constrain rates in numba njit decorator
        Parameters
        ----------
        omega_collection : numpy.ndarray
            2D (dim, blocksize) array containing data with `float` type.
        indices : numpy.ndarray
            1D array containing the index of constraining nodes
        """

        block_size = indices.size
        for i in range(block_size):
            k = indices[i]
            omega_collection[0, k] = 0.0
            omega_collection[1, k] = 0.0
            omega_collection[2, k] = 0.0