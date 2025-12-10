import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb


def plot_video(
    plot_params_rod1: dict,
    plot_params_rod2: dict,
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

            plt.xlim(xlim / unit_scaling)
            plt.ylim(ylim / unit_scaling)

            plt.xlabel("x (" + unit + ")")
            plt.ylabel("y (" + unit + ")")

            writer.grab_frame()



def plot_end_position_vs_time(
    plot_params_rod1: dict,
    ylim: list,
    tlim: list,
    unit="m",
    plot_name="plot_end_pos_time.png",
):
    import matplotlib.pyplot as plt

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
    plt.savefig(plot_name)
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
    plt.savefig(plot_name)
    # plt.show()
    plt.close(fig)



def plot_end_acceleration_vs_time(
    plot_params_rod1: dict,
    tlim: list,
    plot_name="plot_end_accel_time.png",
):
    import matplotlib.pyplot as plt

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
    plt.savefig(plot_name)
    # plt.show()
    plt.close(fig)



def plot_force_vs_displacement(
    plot_params_rod1: dict,
    ylim: list,
    ref_y=0.0,
    unit="m",
    plot_name="plot_force_disp.png",
):
    import matplotlib.pyplot as plt

    if(unit=="mm"):
        unit_scaling = 0.001 # plot in mm
    elif(unit=="m"):
        unit_scaling = 1

    force = np.array(plot_params_rod1["force"])
    position_of_rod1 = np.array(plot_params_rod1["position"]) / unit_scaling
    ref_y = ref_y / unit_scaling

    fig = plt.figure()
    plt.plot(
        ref_y - position_of_rod1[:, 1, -1], 
        -force[:, 1, -1],
    )
    
    plt.xlim(ylim / unit_scaling)

    plt.xlabel("key displacement (" + unit + ")")
    plt.ylabel("force (N)")
    plt.savefig(plot_name)
    # plt.show()
    plt.close(fig)