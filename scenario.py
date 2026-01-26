import numpy as np
from datetime import datetime
from ordered_set import OrderedSet
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

def make_timesteps(cfg):
    start_time = datetime.now().replace(microsecond=0)
    return [start_time + k * cfg.dt for k in range(cfg.n_steps)]

def make_truths(cfg, timesteps):
    truths = OrderedSet()
    time_to_collision = cfg.collision_step * cfg.dt.total_seconds()
    start_distance = cfg.speed * time_to_collision

    for i in range(cfg.num_targets):
        angle = np.radians(i * (360 / cfg.num_targets))
        x0 = cfg.sensor_x + start_distance * np.cos(angle)
        y0 = cfg.sensor_y + start_distance * np.sin(angle)

        turn_rate = np.radians(0.5 if i % 2 == 0 else -0.5)
        heading = angle + np.pi - (turn_rate * time_to_collision / 2)

        vx = cfg.speed * np.cos(heading)
        vy = cfg.speed * np.sin(heading)

        path = GroundTruthPath([
            GroundTruthState([x0, vx, y0, vy], timestamp=timesteps[0])
        ])

        cx, cy, cvx, cvy = x0, y0, vx, vy

        for k in range(1, cfg.n_steps):
            ca = np.cos(turn_rate * cfg.dt.total_seconds())
            sa = np.sin(turn_rate * cfg.dt.total_seconds())

            cvx, cvy = cvx * ca - cvy * sa, cvx * sa + cvy * ca
            cx += cvx * cfg.dt.total_seconds()
            cy += cvy * cfg.dt.total_seconds()

            path.append(GroundTruthState([cx, cvx, cy, cvy], timestamp=timesteps[k]))

        truths.add(path)

    return truths
