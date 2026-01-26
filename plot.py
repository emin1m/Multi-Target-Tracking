from stonesoup.plotter import AnimatedPlotterly

def show_animation(timesteps, truths, measurements, tracks):
    plotter = AnimatedPlotterly(timesteps, tail_length=1)
    plotter.plot_ground_truths(truths, [0, 2])
    plotter.plot_measurements(measurements, [0, 2])
    plotter.plot_tracks(tracks, [0, 2], uncertainty=True)
    plotter.fig.show()
