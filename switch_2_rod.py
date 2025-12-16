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
    plot_preload,
    fitness,
)
from dynamic_end_force import EndpointForcesFeedback
from switch_callback import SwitchCallBack

np.set_printoptions(precision=5, suppress=False, threshold=np.inf)
class SwitchSimulator(
    ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.Damping, ea.Connections, ea.CallBacks
):
    pass

def run_sim(radius, height, x1, y1, x3, y3, x4, y4, do_plots=False,):
    preload_sim = SwitchSimulator()
    switch_sim = SwitchSimulator()
    mm = 0.001
    if (radius <= 0 or height <= 0): # To prevent CMA from taking a negative radius or height for some reason as the guess
        return 999999999999999


    # simulation parameters
    preload_time = 3
    preload_damping_constant = 0.01
    preload_time_step = 1e-4
    preload_steps = int(preload_time / preload_time_step)

    final_time = 15
    damping_constant = 0.2
    time_step = 1e-4
    total_steps = int(final_time / time_step)
    rendering_fps = 15
    step_skip = int(1.0 / (rendering_fps * time_step))

    n_elem = 20

    # plotting
    xlim = np.array([-80, 80]) * mm
    ylim = np.array([-40, 40]) * mm
    plot_units = "mm"
    plot_target_force_disp = True

    # rod parameters
    base_radius = radius * mm # TODO optimize this
    initial_height = height * mm # TODO optimize this

    origin = np.zeros((3,))
    point_1 = np.array([x1, y1, 0.0]) * mm # TODO optimize this (x and y only)
    point_2 = np.array([0.0, initial_height, 0.0])
    point_3 = np.array([x3, y3, 0.0]) * mm # TODO optimize this (x and y only)
    point_4 = np.array([x4, y4, 0.0]) * mm # TODO optimize this (x and y only)

    base_area = np.pi * (base_radius**2)
    density = 80000 # kg/m^3
    E = 5e5 # Pa
    poisson_ratio = 0.5
    shear_modulus = E / (poisson_ratio + 1.0)

    max_displacement = 14.0 * mm

    third_rod = True
    orientation_boundary_condition = False

    n_node = n_elem + 1

    velocity_target=np.array([0.0, -max_displacement/final_time, 0.0])
    preload_force = np.array([0.0, -0.5, 0.0])

    length_1 = np.linalg.norm(point_2 - point_1)
    length_2 = np.linalg.norm(point_2 - point_3)
    length_3 = np.linalg.norm(point_2 - point_4)

    # initial directors
    direction_1 = (point_2 - point_1) / np.linalg.norm(point_2 - point_1)
    direction_2 = (point_2 - point_3) / np.linalg.norm(point_2 - point_3)
    direction_3 = (point_2 - point_4) / np.linalg.norm(point_2 - point_4)
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
    )
    rod_3 = ea.CosseratRod.straight_rod(
        n_elem,
        point_4,
        direction_3,
        normal,
        length_3,
        base_radius,
        density,
        youngs_modulus=E,
        shear_modulus=shear_modulus,
    )

    preload_sim.append(rod_1)
    preload_sim.append(rod_2)
    if (third_rod): preload_sim.append(rod_3)

    preload_sim.dampen(rod_1).using(
        ea.AnalyticalLinearDamper,
        damping_constant=preload_damping_constant,
        time_step=preload_time_step,
    )
    preload_sim.dampen(rod_2).using(
        ea.AnalyticalLinearDamper,
        damping_constant=preload_damping_constant,
        time_step=preload_time_step,
    )
    if (third_rod): preload_sim.dampen(rod_3).using(
        ea.AnalyticalLinearDamper,
        damping_constant=preload_damping_constant,
        time_step=preload_time_step,
    )

    preload_sim.add_forcing_to(rod_1).using(ea.EndpointForces, 0.0 * preload_force, preload_force, ramp_up_time=0.2)

    if (orientation_boundary_condition):
        preload_sim.constrain(rod_1).using(ea.FixedConstraint, constrained_position_idx=(0,), constrained_director_idx=(0,-1))
        preload_sim.constrain(rod_2).using(ea.FixedConstraint, constrained_position_idx=(0,), constrained_director_idx=(0,-1))
        if (third_rod): preload_sim.constrain(rod_3).using(
            ea.FixedConstraint, constrained_position_idx=(0,), constrained_director_idx=(0,-1))
    else:
        preload_sim.constrain(rod_1).using(ea.FixedConstraint, constrained_position_idx=(0,))
        preload_sim.constrain(rod_2).using(ea.FixedConstraint, constrained_position_idx=(0,))
        if (third_rod): preload_sim.constrain(rod_3).using(
            ea.FixedConstraint, constrained_position_idx=(0,))

    preload_sim.connect(
        first_rod=rod_1, second_rod=rod_2, first_connect_idx=-1, second_connect_idx=-1
    ).using(
        ea.HingeJoint, k=1e5, nu=0, kt=1e1, normal_direction=hinge_axis
    )
    if (third_rod): preload_sim.connect(
        first_rod=rod_1, second_rod=rod_3, first_connect_idx=-1, second_connect_idx=-1
    ).using(
        ea.HingeJoint, k=1e5, nu=0, kt=1e1, normal_direction=hinge_axis
    )

    preload_history: dict[str, list] = defaultdict(list)
    preload_sim.collect_diagnostics(rod_1).using(
        SwitchCallBack, step_skip=step_skip, callback_params=preload_history
    )

    preload_sim.finalize()
    timestepper: ea.typing.StepperProtocol = ea.PositionVerlet()
    time = ea.integrate(timestepper, preload_sim, preload_time, preload_steps)
    ea.save_state(preload_sim, "save_states/", time, True)

    plot_preload(
        preload_history,
        ylim=ylim,
        unit="mm",
        tlim=[0,preload_time+0.2],
    )

    switch_sim.append(rod_1)
    switch_sim.append(rod_2)
    if (third_rod): switch_sim.append(rod_3)

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
    if (third_rod): switch_sim.dampen(rod_3).using(
        ea.AnalyticalLinearDamper,
        damping_constant=damping_constant,
        time_step=time_step,
    )

    if (orientation_boundary_condition):
        switch_sim.constrain(rod_1).using(ea.FixedConstraint, constrained_position_idx=(0,), constrained_director_idx=(0,-1))
        switch_sim.constrain(rod_2).using(ea.FixedConstraint, constrained_position_idx=(0,), constrained_director_idx=(0,-1))
        if (third_rod): switch_sim.constrain(rod_3).using(
            ea.FixedConstraint, constrained_position_idx=(0,), constrained_director_idx=(0,-1))
    else:
        switch_sim.constrain(rod_1).using(ea.FixedConstraint, constrained_position_idx=(0,))
        switch_sim.constrain(rod_2).using(ea.FixedConstraint, constrained_position_idx=(0,))
        if (third_rod): switch_sim.constrain(rod_3).using(
            ea.FixedConstraint, constrained_position_idx=(0,))

    switch_sim.connect(
        first_rod=rod_1, second_rod=rod_2, first_connect_idx=-1, second_connect_idx=-1
    ).using(
        ea.HingeJoint, k=1e5, nu=0, kt=1e1, normal_direction=hinge_axis
    )
    if (third_rod): switch_sim.connect(
        first_rod=rod_1, second_rod=rod_3, first_connect_idx=-1, second_connect_idx=-1
    ).using(
        ea.HingeJoint, k=1e5, nu=0, kt=1e1, normal_direction=hinge_axis
    )

    # system.external_forces[] is our applied force + all the forces associated with joints, 
    # so we can't use it for our force measure! (we want applied force only)
    # we need this list instead, which gets modified by the EndpointForcesFeedback class
    # we also cannot use a callback, since force_measure is not a variable in the RodType class
    force_measure = np.zeros(int(total_steps/step_skip)+1) 
    switch_sim.add_forcing_to(rod_1).using(
        EndpointForcesFeedback, velocity_target=velocity_target, step_skip=step_skip,
        ramp_up_time=0.5, time_step=time_step, force_measure=force_measure[...], preload_force=preload_force
    )

    ea.load_state(switch_sim, "save_states/", True)

    recorded_history_1: dict[str, list] = defaultdict(list)
    recorded_history_2: dict[str, list] = defaultdict(list)
    recorded_history_3: dict[str, list] = defaultdict(list)

    switch_sim.collect_diagnostics(rod_1).using(
        SwitchCallBack, step_skip=step_skip, callback_params=recorded_history_1
    )
    switch_sim.collect_diagnostics(rod_2).using(
        SwitchCallBack, step_skip=step_skip, callback_params=recorded_history_2
    )
    if (third_rod): switch_sim.collect_diagnostics(rod_3).using(
        SwitchCallBack, step_skip=step_skip, callback_params=recorded_history_3
    )

    switch_sim.finalize()
    ea.integrate(timestepper, switch_sim, final_time, total_steps)

    force_measure[-1] = force_measure[-2]
    recorded_history_1["force_measure"] = force_measure
    
    if do_plots:
        plot_end_position_vs_time(
            recorded_history_1,
            ylim=ylim,
            unit="mm",
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
            ylim=[0,max_displacement],
            ref_y=initial_height,
            unit="mm",
            show_target=plot_target_force_disp,
        )
        plot_video(
            recorded_history_1,
            recorded_history_2,
            recorded_history_3,
            video_name="switch.mp4",
            fps=rendering_fps,
            unit=plot_units,
            xlim=xlim,
            ylim=ylim,
        )

    simulated_disp = recorded_history_1["position"][:, 1]
    simulated_force = recorded_history_1["force_measure"]

    score = fitness(simulated_disp, simulated_force)

    return score

# print(recorded_history_1["force"][-1])