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

    time = plot_params_rod1["time"]

    if(unit=="mm"):
        unit_scaling = 0.001 # plot in mm
    elif(unit=="m"):
        unit_scaling = 1

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

# def plot_end_position_vs_time(
#     plot_params_rod1: dict,
#     ylim: list,
#     tlim: list,
#     unit="m",
#     plot_name="plot_end_pos_time.png",
#     fps=15,
# ):
#     import matplotlib.pyplot as plt

#     position_of_rod1[time, 0], 
