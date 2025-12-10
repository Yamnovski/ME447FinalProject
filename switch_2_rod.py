import numpy as np
from matplotlib import pyplot as plt
import vapory as vp
import elastica as ea
from collections import defaultdict
from switch_postprocessing import (
    plot_video,
    plot_end_position_vs_time,
    plot_end_velocity_vs_time,
    plot_end_acceleration_vs_time,
    plot_force_vs_displacement,
)
from dynamic_end_force import EndpointForcesFeedback

# np.set_printoptions(precision=5, suppress=True, threshold=6) # TODO
np.set_printoptions(precision=5, suppress=False, threshold=np.inf) # TODO


class SwitchSimulator(
    ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.Damping, ea.Connections, ea.CallBacks
):
    pass

switch_sim = SwitchSimulator()

mm = 0.001


# simulation parameters
final_time = 10
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

initial_height = 100.0 * mm

origin = np.zeros((3,))
point_1 = np.array([-300.0, 0.0, 0.0]) * mm
point_2 = np.array([0.0, initial_height, 0.0]) # connection point
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
    EndpointForcesFeedback, velocity_target=np.array([0.0, -0.02, 0.0]), ramp_up_time=0.1, time_step=time_step
)

switch_sim.constrain(rod_1).using(
    ea.FixedConstraint,
    constrained_position_idx=(0,),
    constrained_director_idx=(0,-1),
)
switch_sim.constrain(rod_2).using(
    ea.FixedConstraint,
    constrained_position_idx=(0,),
    constrained_director_idx=(0,-1),
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
            self.callback_params["velocity"].append(system.velocity_collection.copy())
            self.callback_params["acceleration"].append(system.acceleration_collection.copy())
            self.callback_params["force"].append(system.external_forces.copy())

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


plot_end_position_vs_time(
    recorded_history_1,
    ylim=ylim,
    tlim=[0,final_time+1],
)

plot_end_velocity_vs_time(
    recorded_history_1,
    tlim=[0,final_time+1],
)

plot_end_acceleration_vs_time(
    recorded_history_1,
    tlim=[0,final_time+1],
)

plot_force_vs_displacement(
    recorded_history_1,
    ylim=ylim,
    ref_y=initial_height,
)

plot_video(
    recorded_history_1,
    recorded_history_2,
    video_name="switch.mp4",
    fps=rendering_fps,
    unit=plot_units,
    xlim=xlim,
    ylim=ylim,
)

# print(recorded_history_1["force"][-1])