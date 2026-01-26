import numpy as np
from stonesoup.types.detection import TrueDetection, Clutter
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange

def build_measurement_model(cfg):
    return CartesianToBearingRange(
        ndim_state=5,
        mapping=(0, 2),
        noise_covar=np.diag([
            cfg.bearing_noise_std**2,
            cfg.range_noise_std**2
        ]),
        translation_offset=np.array([[cfg.sensor_x], [cfg.sensor_y]])
    )

def generate_measurements(cfg, truths, timesteps, measurement_model):
    time_to_collision = cfg.collision_step * cfg.dt.total_seconds()
    start_distance = cfg.speed * time_to_collision

    range_max = start_distance + 5000.0
    clutter_area = np.pi * (range_max**2)
    clutter_density = cfg.mean_clutter / clutter_area

    all_measurements = []

    for k, scan_time in enumerate(timesteps):
        meas_set = set()

        # True detections
        for truth in truths:
            if np.random.rand() <= cfg.prob_detect:
                z = measurement_model.function(truth[k], noise=True)
                meas_set.add(TrueDetection(
                    z,
                    timestamp=scan_time,
                    measurement_model=measurement_model,
                    groundtruth_path=truth
                ))

        # Clutter
        n_clutter = np.random.poisson(cfg.mean_clutter)
        for _ in range(n_clutter):
            bear = np.random.uniform(-np.pi, np.pi)
            rng = np.random.uniform(0, range_max)
            z_clut = np.array([[bear], [rng]])
            meas_set.add(Clutter(
                z_clut,
                timestamp=scan_time,
                measurement_model=measurement_model
            ))

        all_measurements.append(meas_set)

    return all_measurements, clutter_density
