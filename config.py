from dataclasses import dataclass
from datetime import timedelta
import numpy as np

@dataclass
class Config:
    seed: int = 777

    # Simulation
    n_steps: int = 90
    dt: timedelta = timedelta(seconds=2)

    # Sensor
    sensor_x: float = 0.0
    sensor_y: float = 0.0

    # Targets
    num_targets: int = 6
    speed: float = 300.0
    collision_step: int = 45

    # Detection / clutter
    mean_clutter: float = 10.0
    prob_detect: float = 0.90
    prob_gate: float = 0.90
    max_missed: int = 6

    # Measurement noise
    range_noise_std: float = 150.0
    bearing_noise_std: float = np.radians(1)

    # Process noise
    q_std: float = 10.0
    turn_noise_std: float = np.radians(0.1)

    # Track management
    confirmation_threshold: int = 10

    # Initialisation noise
    init_pos_std: float = 500.0
    init_vel_std: float = 100.0
