import elastica as ea

import numpy as np
from numpy.typing import NDArray

from elastica.typing import SystemType, RodType, RigidBodyType
from elastica import NoForces

class EndpointForcesFeedback(NoForces):
    """
    This class applies dynamic force on the endpoint node to approach a constant target endpoint velocity.

        Attributes
        ----------
        kp: float
            proportional constant for feedback control
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.

    """

    def __init__(
        self,
        kp: float,
        ramp_up_time: float,
    ) -> None:
        """

        Parameters
        ----------
        start_force: numpy.ndarray
            1D (dim) array containing data with 'float' type.
            Force applied to first node of the system.
        end_force: numpy.ndarray
            1D (dim) array containing data with 'float' type.
            Force applied to last node of the system.
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.

        """
        super(EndpointForcesFeedback, self).__init__()
        self.kp = kp
        assert ramp_up_time > 0.0
        self.ramp_up_time = np.float64(ramp_up_time)

    def apply_forces(
        self, system: "RodType | RigidBodyType", time: np.float64 = np.float64(0.0)
    ) -> None:
        self.compute_end_point_forces(
            system.external_forces,
            self.kp,
            time,
            self.ramp_up_time,
        )



    @staticmethod
    @njit(cache=True)  # type: ignore
    def compute_end_point_forces(
        external_forces: NDArray[np.float64],
        start_force: NDArray[np.float64],
        end_force: NDArray[np.float64],
        time: np.float64,
        ramp_up_time: np.float64,
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
        factor = min(1.0, float(time / ramp_up_time))
        external_forces[..., 0] += start_force * factor
        external_forces[..., -1] += end_force * factor