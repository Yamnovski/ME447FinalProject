import numpy as np
from matplotlib import pyplot as plt
import vapory as vp
import os
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
from buckle_bc import BuckleBC
from dynamic_end_force import EndpointForcesFeedback
from switch_callback import SwitchCallBack

np.set_printoptions(precision=5, suppress=False, threshold=np.inf)
class SwitchSimulator(
    ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.Damping, ea.Connections, ea.CallBacks
):
    pass

def run_sim(r1, r2, r3, rod_4_stiff, preload, x1, y1, x3, y3, x4, y4, orientation_boundary_condition, do_plots=False,):

    preload_sim = SwitchSimulator()
    switch_sim = SwitchSimulator()
    mm = 0.001

    # simulation parameters
    preload_time = 3
    preload_damping_constant = 0.01
    preload_time_step = 1e-4
    preload_steps = int(preload_time / preload_time_step)

    final_time = 25
    damping_constant = 0.2
    time_step = 1e-4
    total_steps = int(final_time / time_step)
    rendering_fps = 10
    step_skip = int(1.0 / (rendering_fps * time_step))

    n_elem = 20

    # plotting
    xlim = np.array([-90, 90]) * mm
    ylim = np.array([-45, 45]) * mm
    plot_units = "mm"
    plot_target_force_disp = True

    # rod parameters
    radius_1 = np.abs(r1) * mm # TODO optimize this
    radius_2 = np.abs(r2) * mm # TODO optimize this
    radius_3 = np.abs(r3) * mm # TODO optimize this
    radius_4 = 3.3 * mm # TODO optimize this
    initial_height = 0.0 * mm

    origin = np.zeros((3,))
    point_1 = np.array([x1, y1, 0.0]) * mm # TODO optimize this (x and y only)
    point_2 = np.array([0.0, initial_height, 0.0])
    point_3 = np.array([x3, y3, 0.0]) * mm # TODO optimize this (x and y only)
    point_4 = np.array([x4, y4, 0.0]) * mm # TODO optimize this (x and y only)

    point_5 = np.array([0.0, -35.5, 0.0]) * mm # this only affects the visual indication of where the buckle rod is

    density = 80000 # kg/m^3 
    # ^ this is not physically accurate, but it shouldn't matter much for quasi-static simulation, and it prevents blowing up

    E = 5e5 # Pa
    poisson_ratio = 0.5
    shear_modulus = E / (poisson_ratio + 1.0)

    max_displacement = 14.0 * mm
    contact_start_displacement = 0.0 * mm
    rod_4_stiffness_factor = rod_4_stiff / 100 # TODO optimize this (100 factor is simple initial sigma adjustment for CMA)

    third_rod = True
    fourth_rod = True

    n_node = n_elem + 1

    velocity_target=np.array([0.0, -max_displacement/final_time, 0.0])
    preload_force = np.array([0.0, -np.clip(preload / 10, min=0.0, max=0.5), 0.0]) # sigma adjustment factor for preload

    length_1 = np.linalg.norm(point_2 - point_1)
    length_2 = np.linalg.norm(point_2 - point_3)
    length_3 = np.linalg.norm(point_2 - point_4)
    length_4 = 31.0 * mm 

    # initial directors
    direction_1 = (point_2 - point_1) / np.linalg.norm(point_2 - point_1)
    direction_2 = (point_2 - point_3) / np.linalg.norm(point_2 - point_3)
    direction_3 = (point_2 - point_4) / np.linalg.norm(point_2 - point_4)
    direction_4 = np.array([0.0, 1.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])

    hinge_axis = np.array([0.0, 0.0, 1.0])

    rod_1 = ea.CosseratRod.straight_rod(
        n_elem,
        point_1,
        direction_1,
        normal,
        length_1,
        radius_1,
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
        radius_2,
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
        radius_3,
        density,
        youngs_modulus=E,
        shear_modulus=shear_modulus,
    )
    rod_4 = ea.CosseratRod.straight_rod(
        n_elem,
        point_5,
        direction_4,
        normal,
        length_4,
        radius_4,
        density,
        youngs_modulus=E,
        shear_modulus=shear_modulus,
    )

    preload_sim.append(rod_1)
    preload_sim.append(rod_2)
    if (third_rod): preload_sim.append(rod_3)
    if (fourth_rod): preload_sim.append(rod_4)

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
    if (fourth_rod): preload_sim.dampen(rod_4).using(
        ea.AnalyticalLinearDamper,
        damping_constant=preload_damping_constant,
        time_step=preload_time_step,
    )

    preload_sim.add_forcing_to(rod_1).using(ea.EndpointForces, 0.0 * preload_force, preload_force, ramp_up_time=0.2)

    if (orientation_boundary_condition):
        preload_sim.constrain(rod_1).using(
            ea.FixedConstraint, constrained_position_idx=(0,), constrained_director_idx=(0,-1))
        preload_sim.constrain(rod_2).using(
            ea.FixedConstraint, 
            constrained_position_idx=(0,), 
            constrained_director_idx=(0,-1))
        if (third_rod): preload_sim.constrain(rod_3).using(
            ea.FixedConstraint, constrained_position_idx=(0,), constrained_director_idx=(0,-1))
    else:
        preload_sim.constrain(rod_1).using(ea.FixedConstraint, constrained_position_idx=(0,))
        preload_sim.constrain(rod_2).using(ea.FixedConstraint, constrained_position_idx=(0,))
        if (third_rod): preload_sim.constrain(rod_3).using(ea.FixedConstraint, constrained_position_idx=(0,))
        
    if (fourth_rod): preload_sim.constrain(rod_4).using(BuckleBC, constrained_position_idx=(0,-1))


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
    time = ea.integrate(timestepper, preload_sim, preload_time, preload_steps, progress_bar=False)
    ea.save_state(preload_sim, "save_states/", time)

    if (np.isnan(preload_history["position"]).any()): # catch a simulation that blew up before running the slow simulation
        return 999999999999999

    plot_preload(
        preload_history,
        ylim=ylim,
        unit="mm",
        tlim=[0,preload_time+0.2],
    )

    switch_sim.append(rod_1)
    switch_sim.append(rod_2)
    if (third_rod): switch_sim.append(rod_3)
    if (fourth_rod): switch_sim.append(rod_4)


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
    if (fourth_rod): switch_sim.dampen(rod_4).using(
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

    if (fourth_rod): switch_sim.constrain(rod_4).using(BuckleBC, constrained_position_idx=(0,-1))


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

    force_measure = np.zeros(int(total_steps/step_skip)+1) 
    force_measure_2 = np.zeros(int(total_steps/step_skip)+1) 
    buckle_force = np.zeros(3, dtype=np.float64)

    # system.external_forces[] is our applied force + all the forces associated with joints, 
    # so we can't use it for our force measure! (we want applied force only)
    # we need this list instead, which gets modified by the EndpointForcesFeedback class
    # we also cannot use a callback, since force_measure is not a variable in the RodType class
    
    if (fourth_rod): switch_sim.add_forcing_to(rod_4).using(
        EndpointForcesFeedback, type="buckle", velocity_target=velocity_target, step_skip=step_skip,
        ramp_up_time=0.5, time_step=time_step, force_measure=force_measure_2.view(), preload_force=np.zeros(3),
        get_buckle_force=buckle_force.view(), time_delay=final_time * contact_start_displacement / max_displacement,
    )
    if (fourth_rod): switch_sim.add_forcing_to(rod_4).using(
        ea.UniformForces, force=0.5, direction=np.array([1.0, 0.0, 0.0]))

    switch_sim.add_forcing_to(rod_1).using(
        EndpointForcesFeedback, type="main", velocity_target=velocity_target, step_skip=step_skip,
        ramp_up_time=0.5, time_step=time_step, force_measure=force_measure.view(), preload_force=preload_force, 
        get_buckle_force=buckle_force.view(), buckle_stiffness_factor=rod_4_stiffness_factor,
    )


    ea.load_state(switch_sim, "save_states/")

    recorded_history_1: dict[str, list] = defaultdict(list)
    recorded_history_2: dict[str, list] = defaultdict(list)
    recorded_history_3: dict[str, list] = defaultdict(list)
    recorded_history_4: dict[str, list] = defaultdict(list)

    switch_sim.collect_diagnostics(rod_1).using(
        SwitchCallBack, step_skip=step_skip, callback_params=recorded_history_1
    )
    switch_sim.collect_diagnostics(rod_2).using(
        SwitchCallBack, step_skip=step_skip, callback_params=recorded_history_2
    )
    if (third_rod): switch_sim.collect_diagnostics(rod_3).using(
        SwitchCallBack, step_skip=step_skip, callback_params=recorded_history_3
    )
    if (fourth_rod): switch_sim.collect_diagnostics(rod_4).using(
        SwitchCallBack, step_skip=step_skip, callback_params=recorded_history_4
    )
        
    # buckling_force = 2.0
    # switch_sim.add_forcing_to(rod_4).using(
    #     ea.EndpointForces, 0.0 * direction_4, np.array([0.0, -buckling_force, 0.0]), ramp_up_time=0.2)


    switch_sim.finalize()
    ea.integrate(timestepper, switch_sim, final_time, total_steps)

    force_measure[-1] = force_measure[-2]
    if (fourth_rod): recorded_history_1["force_measure"] = force_measure

    force_measure_2[-1] = force_measure_2[-2]
    if (fourth_rod): recorded_history_4["force_measure"] = force_measure_2
    
    plot_force_vs_displacement(
        recorded_history_1,
        ylim=[0,max_displacement],
        ref_y=initial_height,
        unit="mm",
        show_target=plot_target_force_disp,
    )

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
        plot_video(
            recorded_history_1,
            recorded_history_2,
            recorded_history_3,
            recorded_history_4,
            video_name="switch.mp4",
            fps=rendering_fps,
            unit=plot_units,
            xlim=xlim,
            ylim=ylim,
        )

    

    score = fitness(recorded_history_1, ref_y=initial_height)

    return score

# print(recorded_history_1["force"][-1])