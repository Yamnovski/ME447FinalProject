import elastica as ea

import numpy as np
from numpy.typing import NDArray

from elastica.typing import SystemType, RodType, RigidBodyType
from elastica import NoForces

from numba import njit

class EndpointForcesFeedback(NoForces):
    """
    This class applies dynamic force on the endpoint node to approach a constant target endpoint velocity.

        Attributes
        ----------
        velocity_target: np.array
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.

    """

    def __init__(
        self,
        velocity_target: NDArray[np.float64],
        ramp_up_time: float,
        time_step: float,
    ) -> None:
        
        super(EndpointForcesFeedback, self).__init__()
        self.velocity_target = velocity_target
        assert ramp_up_time > 0.0
        self.ramp_up_time = np.float64(ramp_up_time)

        self.factor = 0.0
        self.error = np.zeros(3, dtype=np.float64)
        self.error_integrator = np.zeros(3, dtype=np.float64)

        self.time_step = time_step

    def apply_forces(
        self, system: "RodType | RigidBodyType", time: np.float64 = np.float64(0.0)
    ) -> None:
        self.factor = min(1.0, float(time / self.ramp_up_time))
        self.error = (self.velocity_target - system.velocity_collection[..., -1]) * self.factor
        self.error_integrator += self.error * self.time_step

        self.compute_end_point_force(
            system.external_forces,
            time,
            self.ramp_up_time,
            self.error,
            self.error_integrator,
        )



    @staticmethod
    @njit(cache=True)  # type: ignore
    def compute_end_point_force(
        external_forces: NDArray[np.float64],
        time: np.float64,
        ramp_up_time: np.float64,
        error: NDArray[np.float64],
        error_integrator: NDArray[np.float64],
    ) -> None:
        """
        Compute end point forces that are applied on the rod using numba njit decorator.

        Parameters
        ----------
        external_forces: numpy.ndarray
            2D (dim, blocksize) array containing data with 'float' type. External force vector.
        start_force: numpy.ndarray
            1D (dim) array containing data with 'float' type.
        end_force: numpy.ndarray
            1D (dim) array containing data with 'float' type.
            Force applied to last node of the system.
        time: float
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.

        """
        kp = 0.2
        ki = 4000.0

        force = kp * error + ki * error_integrator

        external_forces[..., -1] += force