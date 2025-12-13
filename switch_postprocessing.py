import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb


def target_force(displacement):
    target_force_data = np.genfromtxt('target_force_data.csv', delimiter=',', dtype=np.float64)
    displacements = target_force_data[:,0] * 0.0254 # convert inches to m
    forces = target_force_data[:,1] * 0.0098 # convert g to N
    return(np.interp(displacement, displacements, forces))


def plot_video(
    plot_params_rod1: dict,
    plot_params_rod2: dict,
    plot_params_rod3: dict,
    xlim: list,
    ylim: list,
    unit="m",
    video_name="video.mp4",
    fps=15,
):  # (time step, x/y/z, node)
    import matplotlib.animation as manimation

    if(unit=="mm"):
        unit_scaling = 0.001 # plot in mm
    elif(unit=="m"):
        unit_scaling = 1

    time = plot_params_rod1["time"]
    position_of_rod1 = np.array(plot_params_rod1["position"]) / unit_scaling
    position_of_rod2 = np.array(plot_params_rod2["position"]) / unit_scaling

    
    if (len(plot_params_rod3) != 0):
        position_of_rod3 = np.array(plot_params_rod3["position"]) / unit_scaling
        third_rod = True
    else: 
        third_rod = False

    print("plot video xy")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure()
    plt.axis("equal")
    with writer.saving(fig, video_name, 100):
        for time in range(1, len(time)):
            fig.clf()
            plt.plot(
                position_of_rod1[time, 0], 
                position_of_rod1[time, 1], 
                c=to_rgb("xkcd:red"),
                label="rod1"
            )
            plt.plot(
                position_of_rod2[time, 0],
                position_of_rod2[time, 1],
                c=to_rgb("xkcd:bluish"),
                label="rod2",
            )
            if (third_rod): plt.plot(
                position_of_rod3[time, 0],
                position_of_rod3[time, 1],
                c=to_rgb("xkcd:green"),
                label="rod3",
            )

            plt.xlim(xlim / unit_scaling)
            plt.ylim(ylim / unit_scaling)

            plt.xlabel("x (" + unit + ")")
            plt.ylabel("y (" + unit + ")")

            writer.grab_frame()


def plot_preload(
    plot_params_rod1: dict,
    ylim: list,
    tlim: list,
    unit="m",
    plot_name="plot_preload.png",
):
    if(unit=="mm"):
        unit_scaling = 0.001 # plot in mm
    elif(unit=="m"):
        unit_scaling = 1

    time = plot_params_rod1["time"]
    position_of_rod1 = np.array(plot_params_rod1["position"]) / unit_scaling

    fig = plt.figure()
    plt.plot(
        time,
        position_of_rod1[:, 1, -1], 
    )
    plt.xlim(tlim)
    plt.ylim(ylim / unit_scaling)

    plt.xlabel("time (s)")
    plt.ylabel("position (" + unit + ")")
    plt.savefig(plot_name, bbox_inches='tight')
    # plt.show()
    plt.close(fig)


def plot_end_position_vs_time(
    plot_params_rod1: dict,
    ylim: list,
    tlim: list,
    unit="m",
    plot_name="plot_end_pos_time.png",
):
    if(unit=="mm"):
        unit_scaling = 0.001 # plot in mm
    elif(unit=="m"):
        unit_scaling = 1

    time = plot_params_rod1["time"]
    position_of_rod1 = np.array(plot_params_rod1["position"]) / unit_scaling

    fig = plt.figure()
    plt.plot(
        time,
        position_of_rod1[:, 1, -1], 
    )
    plt.xlim(tlim)
    plt.ylim(ylim / unit_scaling)

    plt.xlabel("time (s)")
    plt.ylabel("position (" + unit + ")")
    plt.savefig(plot_name, bbox_inches='tight')
    # plt.show()
    plt.close(fig)


def plot_end_velocity_vs_time(
    plot_params_rod1: dict,
    tlim: list,
    plot_name="plot_end_vel_time.png",
):
    import matplotlib.pyplot as plt

    time = plot_params_rod1["time"]
    velocity_of_rod1 = np.array(plot_params_rod1["velocity"])

    fig = plt.figure()
    plt.plot(
        time,
        velocity_of_rod1[:, 1, -1], 
    )
    # plt.plot( # reference/target velocity
    #     time,
    #     np.ones_like(time)*(-0.02), 
    #     linestyle="--",
    # )
    plt.xlim(tlim)
    
    plt.xlabel("time (s)")
    plt.ylabel("velocity (m/s)")
    plt.savefig(plot_name, bbox_inches='tight')
    # plt.show()
    plt.close(fig)


def plot_end_acceleration_vs_time(
    plot_params_rod1: dict,
    tlim: list,
    plot_name="plot_end_accel_time.png",
):
    time = plot_params_rod1["time"]
    acceleration_of_rod1 = np.array(plot_params_rod1["acceleration"])

    fig = plt.figure()
    plt.plot(
        time,
        acceleration_of_rod1[:, 1, -1], 
    )
    plt.xlim(tlim)
    
    plt.xlabel("time (s)")
    plt.ylabel("acceleration (m/s^2)")
    plt.savefig(plot_name, bbox_inches='tight')
    # plt.show()
    plt.close(fig)


def plot_force_vs_displacement(
    plot_params_rod1: dict,
    ylim: list,
    ref_y=0.0,
    unit="m",
    plot_name="plot_force_disp.png",
    show_target=True,
):
    if(unit=="mm"):
        unit_scaling = 0.001 # plot in mm
    elif(unit=="m"):
        unit_scaling = 1

    position_of_rod1 = np.array(plot_params_rod1["position"]) / unit_scaling
    ref_y = ref_y / unit_scaling

    force = -np.array(plot_params_rod1["force_measure"])
    displacement = ref_y - position_of_rod1[:, 1, -1]

    fig = plt.figure()
    plt.plot(
        displacement, 
        force,
    )
    if (show_target):
        plt.plot(
            displacement, 
            target_force((displacement - displacement[0]) * unit_scaling),
            linestyle="--",
        )
    
    # plt.xlim(0, (ylim[1] - ylim[0]) / unit_scaling)
    plt.xlabel("key displacement (" + unit + ")")
    plt.ylabel("force (N)")
    if (show_target):
        plt.legend(["current (fitness: " + str(fitness(plot_params_rod1, ref_y)) + ")", "target"])
        # plt.legend(["current", "target"])
    plt.savefig(plot_name, bbox_inches='tight')
    # plt.show()
    plt.close(fig)


def fitness(
        plot_params_rod1: dict,
        ref_y=0.0,
):
    position_of_rod1 = np.array(plot_params_rod1["position"])
    ref_y = ref_y

    force = -np.array(plot_params_rod1["force_measure"])
    displacement = ref_y - position_of_rod1[:, 1, -1]

    dx = displacement[1:] - displacement[:-1]
    error = 0.5 * (np.abs(force - target_force(displacement))[1:] + np.abs(force - target_force(displacement))[:-1])

    return np.sum(error * dx)