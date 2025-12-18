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
        type: str,
        velocity_target: NDArray[np.float64],
        ramp_up_time: float,
        time_step: float,
        step_skip: float,
        force_measure: NDArray[np.float64],
        get_buckle_force: NDArray[np.float64],
        preload_force: NDArray[np.float64],
        kp = 40,
        ki = 4000.0,
        time_delay = 0.0,
        buckle_stiffness_factor = 0.0,
        
    ) -> None:
        
        super(EndpointForcesFeedback, self).__init__()
        self.velocity_target = velocity_target
        assert ramp_up_time > 0.0
        self.ramp_up_time = np.float64(ramp_up_time)
        self.force_measure = force_measure.view()
        self.get_buckle_force = get_buckle_force.view()
        self.buckle_stiffness_factor = buckle_stiffness_factor
        self.preload_force = preload_force

        self.type = type
        self.step_skip = step_skip
        self.kp = kp
        self.ki = ki
        self.time_delay = time_delay

        self.factor = 0.0
        self.error = np.zeros(3, dtype=np.float64)
        self.error_integrator = self.preload_force / self.ki # start with preload force

        self.time_step = time_step

    def apply_forces(
        self, system: "RodType | RigidBodyType", time: np.float64 = np.float64(0.0)
    ) -> None:
        index = int(np.floor(time/self.time_step/self.step_skip)) # index for force_measure "callback"

        args = (
                system.external_forces,
                time - self.time_delay,
                self.time_step,
                self.ramp_up_time,
                self.velocity_target,
                system.velocity_collection[..., -1],
                self.error_integrator,
                self.kp,
                self.ki,
            )

        if self.type == "main":
            self.force_measure[index] = self.compute_end_point_force(*args)[1]
            system.external_forces[..., -1] += -self.get_buckle_force * self.buckle_stiffness_factor
        elif self.type == "buckle": 
            self.force_measure[index] = self.compute_end_point_force(*args)[1]
            self.get_buckle_force[1] = self.force_measure[index]

        



    @staticmethod
    @njit(cache=True)  # type: ignore
    def compute_end_point_force(
        external_forces: NDArray[np.float64],
        time: np.float64,
        time_step: np.float64,
        ramp_up_time: np.float64,
        velocity_target: NDArray[np.float64],
        current_velocity: NDArray[np.float64],
        error_integrator: NDArray[np.float64],
        kp: np.float64,
        ki: np.float64,
    ) -> None:
        
        factor = max(0.0, min(1.0, float(time / ramp_up_time)))
        error = (velocity_target - current_velocity) * factor
        error_integrator += error * time_step

        force = kp * error + ki * error_integrator # proportional-integral controller maintains small, near-constant velocity
        force[0] = 0.0
        force[2] = 0.0

        external_forces[..., -1] += force

        return force
        