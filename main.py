import numpy as np
from config import Config
from scenario import make_timesteps, make_truths
from measurements import build_measurement_model, generate_measurements
from tracking import run_tracker
from plot import show_animation

def main():
    cfg = Config()
    np.random.seed(cfg.seed)

    timesteps = make_timesteps(cfg)
    truths = make_truths(cfg, timesteps)

    meas_model = build_measurement_model(cfg)
    measurements, clutter_density = generate_measurements(cfg, truths, timesteps, meas_model)

    tracks, total = run_tracker(
        cfg, truths, timesteps, measurements, meas_model, clutter_density
    )

    confirmed = {t for t in tracks if t.metadata.get("confirmed")}
    print(f"TOTAL TRACK WINDOWS: {total}")
    print(f"CONFIRMED TRACKS: {len(confirmed)}")

    show_animation(timesteps, truths, measurements, confirmed)

if __name__ == "__main__":
    main()
