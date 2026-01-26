import numpy as np
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.types.array import StateVectors
from stonesoup.types.update import GaussianStateUpdate
from stonesoup.models.transition.nonlinear import ConstantTurn
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.dataassociator.probability import JPDA
from stonesoup.functions import gm_reduce_single
from stonesoup.types.detection import Clutter

def run_tracker(cfg, truths, timesteps, measurements, measurement_model, clutter_density):
    transition = ConstantTurn(
        linear_noise_coeffs=np.array([cfg.q_std**2, cfg.q_std**2]),
        turn_noise_coeff=cfg.turn_noise_std**2
    )

    predictor = ExtendedKalmanPredictor(transition)
    updater = ExtendedKalmanUpdater(measurement_model)

    associator = JPDA(PDAHypothesiser(
        predictor, updater, clutter_density,
        cfg.prob_detect, cfg.prob_gate
    ))

    tracks, all_tracks, missed = set(), set(), {}
    total_windows = 0

    # Initial tracks
    for gt in truths:
        x = np.vstack((gt[0].state_vector, [0]))
        P = np.diag([
            cfg.init_pos_std**2, cfg.init_vel_std**2,
            cfg.init_pos_std**2, cfg.init_vel_std**2,
            np.radians(1)**2
        ])
        trk = Track([GaussianState(x, P, timesteps[0])])
        trk.metadata.update({"hits": cfg.confirmation_threshold, "confirmed": True})

        tracks.add(trk)
        all_tracks.add(trk)
        missed[trk] = 0
        total_windows += 1

    # Main loop
    for k, meas in enumerate(measurements):
        ts = timesteps[k]
        hyps = associator.associate(tracks, meas, ts)

        for trk in list(tracks):
            states, weights, detected = [], [], False
            for h in hyps[trk]:
                if h:
                    states.append(updater.update(h))
                    detected = True
                else:
                    states.append(h.prediction)
                weights.append(h.probability)

            mean, cov = gm_reduce_single(
                StateVectors([s.state_vector for s in states]),
                np.stack([s.covar for s in states], axis=2),
                np.asarray(weights)
            )

            trk.append(GaussianStateUpdate(mean, cov, hyps[trk], ts))

            if detected:
                missed[trk] = 0
                trk.metadata["hits"] += 1
            else:
                missed[trk] += 1

        for trk in list(tracks):
            if missed[trk] > cfg.max_missed:
                tracks.remove(trk)

    return all_tracks, total_windows
