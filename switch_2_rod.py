import numpy as np
from matplotlib import pyplot as plt
import vapory as vp
import elastica as ea
from collections import defaultdict
from switch_postprocessing import (
    plot_video
)

# np.set_printoptions(precision=5, suppress=True, threshold=6) # TODO
np.set_printoptions(precision=5, suppress=False, threshold=np.inf) # TODO


class SwitchSimulator(
    ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.Damping, ea.Connections, ea.CallBacks
):
    pass

switch_sim = SwitchSimulator()

mm = 0.001


# simulation parameters
final_time = 5
damping_constant = 0.1
time_step = 1e-4
total_steps = int(final_time / time_step)
rendering_fps = 20
step_skip = int(1.0 / (rendering_fps * time_step))

n_elem = 50

# plotting
xlim = np.array([-350, 350]) * mm
ylim = np.array([-350, 350]) * mm
plot_units = "mm"

# rod parameters
base_radius = 10 * mm
base_area = np.pi * (base_radius**2)
density = 1100 # kg/m^3 (TPU plastic)
E = 0.6e6 # Pa (TPU plastic)
poisson_ratio = 0.5
shear_modulus = E / (poisson_ratio + 1.0)

origin = np.zeros((3,))
point_1 = np.array([-300.0, 0.0, 0.0]) * mm
point_2 = np.array([0.0, 50.0, 0.0]) * mm # connection point
point_3 = np.array([300.0, 0.0, 0.0]) * mm

n_node = n_elem + 1

# positions_1 = np.linspace(point_1, point_2, n_node).T
# positions_2 = np.linspace(point_3, point_2, n_node).T

length_1 = np.linalg.norm(point_2 - point_1)
length_2 = np.linalg.norm(point_2 - point_3)

# initial directors
direction_1 = (point_2 - point_1) / np.linalg.norm(point_2 - point_1)
direction_2 = (point_2 - point_3) / np.linalg.norm(point_2 - point_3)
normal = np.array([0.0, 0.0, 1.0])

hinge_axis = np.array([0.0, 0.0, 1.0])

F = np.array([0.0, -0.2, 0.0])

rod_1 = ea.CosseratRod.straight_rod(
    n_elem,
    point_1,
    direction_1,
    normal,
    length_1,
    base_radius,
    density,
    youngs_modulus=E,
    shear_modulus=shear_modulus,
    # position=positions_1
)
rod_2 = ea.CosseratRod.straight_rod(
    n_elem,
    point_3,
    direction_2,
    normal,
    length_2,
    base_radius,
    density,
    youngs_modulus=E,
    shear_modulus=shear_modulus,
    # position=positions_2
)

switch_sim.append(rod_1)
switch_sim.append(rod_2)

switch_sim.dampen(rod_1).using(
    ea.AnalyticalLinearDamper,
    damping_constant=damping_constant,
    time_step=time_step,
)
switch_sim.dampen(rod_2).using(
    ea.AnalyticalLinearDamper,
    damping_constant=damping_constant,
    time_step=time_step,
)

switch_sim.add_forcing_to(rod_1).using(
    ea.EndpointForces, start_force=np.zeros(3), end_force=F, ramp_up_time=1
)

switch_sim.constrain(rod_1).using(
    ea.OneEndFixedBC,
    constrained_position_idx=(0,),
    constrained_director_idx=(0,),
)
switch_sim.constrain(rod_2).using(
    ea.OneEndFixedBC,
    constrained_position_idx=(0,),
    constrained_director_idx=(0,),
)

switch_sim.connect(
    first_rod=rod_1, second_rod=rod_2, first_connect_idx=-1, second_connect_idx=-1
).using(
    ea.HingeJoint, k=1e5, nu=0, kt=1e1, normal_direction=hinge_axis
)

class SwitchCallBack(ea.CallBackBaseClass):
    """
    Call back function for switch
    """

    def __init__(self, step_skip: int, callback_params: dict) -> None:
        super().__init__()
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(
        self, system: ea.typing.RodType, time: np.float64, current_step: int
    ) -> None:

        if current_step % self.every == 0:

            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position"].append(system.position_collection.copy())

            return

recorded_history_1: dict[str, list] = defaultdict(list)
recorded_history_2: dict[str, list] = defaultdict(list)

switch_sim.collect_diagnostics(rod_1).using(
    SwitchCallBack, step_skip=step_skip, callback_params=recorded_history_1
)
switch_sim.collect_diagnostics(rod_2).using(
    SwitchCallBack, step_skip=step_skip, callback_params=recorded_history_2
)

switch_sim.finalize()
timestepper: ea.typing.StepperProtocol = ea.PositionVerlet()
ea.integrate(timestepper, switch_sim, final_time, total_steps)

position = np.array(recorded_history_2["position"])
print(position[-1])

plot_video(
        recorded_history_1,
        recorded_history_2,
        video_name="switch.mp4",
        fps=rendering_fps,
        unit=plot_units,
        xlim=xlim,
        ylim=ylim,
    )
