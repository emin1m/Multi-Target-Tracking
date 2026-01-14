
from datetime import datetime, timedelta
from ordered_set import OrderedSet
import numpy as np

from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.detection import TrueDetection, Clutter
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.types.array import StateVectors
from stonesoup.types.update import GaussianStateUpdate

from stonesoup.models.transition.nonlinear import ConstantTurn 
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange


from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.dataassociator.probability import JPDA
from stonesoup.functions import gm_reduce_single
from stonesoup.plotter import AnimatedPlotterly

np.random.seed(777)

n_steps = 90              
dt = timedelta(seconds=2)   
sensor_x, sensor_y = 0.0, 0.0 


num_targets = 6             
speed = 300.0               
collision_step = 45         
time_to_collision = collision_step * dt.total_seconds()
start_distance = speed * time_to_collision 


mean_clutter = 10.0          
range_noise_std = 150.0        
bearing_noise_std = np.radians(1)


prob_detect = 0.90          
prob_gate = 0.90 
max_missed = 6               

q_std = 10

turn_noise_std = np.radians(0.1) 

CONFIRMATION_THRESHOLD = 10


init_pos_std = 500    
init_vel_std = 100    


total_track_windows = 0

truths = OrderedSet()
start_time = datetime.now().replace(microsecond=0)
timesteps = [start_time + k*dt for k in range(n_steps)]

for i in range(num_targets):
    angle_deg = i * (360 / num_targets)
    angle_rad = np.radians(angle_deg)
    
    x0 = sensor_x + start_distance * np.cos(angle_rad)
    y0 = sensor_y + start_distance * np.sin(angle_rad)

    turn_rate_deg = 0.5 if i % 2 == 0 else -0.5
    turn_rate_rad = np.radians(turn_rate_deg)
    
    total_turn_until_center = turn_rate_rad * time_to_collision
    heading = angle_rad + np.pi - (total_turn_until_center / 2)

    vx = speed * np.cos(heading)
    vy = speed * np.sin(heading)

    gt_path = GroundTruthPath([
        GroundTruthState([x0, vx, y0, vy], timestamp=start_time)
    ])

    curr_x, curr_y = x0, y0
    curr_vx, curr_vy = vx, vy
    
    for k in range(1, n_steps):
        ca = np.cos(turn_rate_rad * dt.total_seconds())
        sa = np.sin(turn_rate_rad * dt.total_seconds())
        
        new_vx = curr_vx * ca - curr_vy * sa
        new_vy = curr_vx * sa + curr_vy * ca
        
        curr_x += new_vx * dt.total_seconds()
        curr_y += new_vy * dt.total_seconds()
        
        curr_vx, curr_vy = new_vx, new_vy
        
        ts = timesteps[k]
        gt_path.append(GroundTruthState([curr_x, curr_vx, curr_y, curr_vy], timestamp=ts))

    truths.add(gt_path)


measurement_model = CartesianToBearingRange(
    ndim_state=5, 
    mapping=(0, 2),
    noise_covar=np.diag([bearing_noise_std**2, range_noise_std**2]),
    translation_offset=np.array([[sensor_x], [sensor_y]])
)

range_max = start_distance + 5000.0
clutter_area = np.pi * (range_max**2)
clutter_density = mean_clutter / clutter_area

all_measurements = []

for k in range(n_steps):
    scan_time = timesteps[k]
    meas_set = set()
    
    for truth in truths:
        if np.random.rand() <= prob_detect:

            z = measurement_model.function(truth[k], noise=True)
            meas_set.add(TrueDetection(
                z, timestamp=scan_time, measurement_model=measurement_model, groundtruth_path=truth
            ))

    n_clutter = np.random.poisson(mean_clutter)
    for _ in range(n_clutter):
        bear = np.random.uniform(-np.pi, np.pi)
        rng = np.random.uniform(0, range_max)
        z_clut = np.array([[bear], [rng]])
        meas_set.add(Clutter(z_clut, timestamp=scan_time, measurement_model=measurement_model))
        
    all_measurements.append(meas_set)


transition_model_tracker = ConstantTurn(
    linear_noise_coeffs=np.array([q_std**2, q_std**2]),  
    turn_noise_coeff=turn_noise_std**2                   
)

predictor = ExtendedKalmanPredictor(transition_model_tracker)
updater = ExtendedKalmanUpdater(measurement_model)

hypothesiser = PDAHypothesiser(
    predictor=predictor,
    updater=updater,
    clutter_spatial_density=clutter_density,
    prob_detect=prob_detect,
    prob_gate=prob_gate
)
associator = JPDA(hypothesiser=hypothesiser)


tracks = set()
all_tracks = set()
missed_counts = {}


for gt in truths:
    state_vec_4d = gt[0].state_vector 
    

    init_vec_5d = np.vstack((state_vec_4d, [0])) 
    

    perturbation = np.array([
        [np.random.normal(0, init_pos_std)],
        [np.random.normal(0, init_vel_std)],
        [np.random.normal(0, init_pos_std)],
        [np.random.normal(0, init_vel_std)],
        [np.random.normal(0, np.radians(0.2))] 
    ])
    
    init_vec = init_vec_5d + perturbation
    
    prior = GaussianState(
        init_vec,
        np.diag([init_pos_std**2, init_vel_std**2, 
                 init_pos_std**2, init_vel_std**2, 
                 np.radians(1.0)**2]),
        timestamp=start_time
    )
    trk = Track([prior])
    
    trk.metadata.update({"hits": CONFIRMATION_THRESHOLD, "confirmed": True}) 
    
    tracks.add(trk)
    all_tracks.add(trk)
    missed_counts[trk] = 0
    total_track_windows += 1

def spawn_tracks(meas_set, curr_tracks, time):
    global total_track_windows
    new = set()
    for m in meas_set:
        if isinstance(m, Clutter): continue 
        
        z = m.state_vector
        mx = sensor_x + z[1,0]*np.cos(z[0,0])
        my = sensor_y + z[1,0]*np.sin(z[0,0])
        
        is_near = False
        for t in curr_tracks:
            tx, ty = t.state.state_vector[0], t.state.state_vector[2]
            if np.hypot(mx-tx, my-ty) < 2500.0: 
                is_near = True
                break
        
        if not is_near:

            prior = GaussianState(
                np.array([[mx], [0], [my], [0], [0]]), 
                np.diag([init_pos_std**2, 100**2, 
                         init_pos_std**2, 100**2, 
                         np.radians(5)**2]), 
                timestamp=time
            )
            new_trk = Track([prior])
            new_trk.metadata.update({"hits": 1, "confirmed": False}) 
            
            new.add(new_trk)
            total_track_windows += 1 
            
    return new


print("Radar Simülasyonu Başlıyor (Model: CONSTANT TURN + JPDA)...")

for n, meas_set in enumerate(all_measurements):
    now = timesteps[n]
    

    if tracks:
        hyps = associator.associate(tracks, meas_set, now)
        for trk in list(tracks):
            h_list = hyps[trk]
            states, weights, detected = [], [], False
            
            for h in h_list:
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
            trk.append(GaussianStateUpdate(mean, cov, h_list, now))
            
            if detected: 
                missed_counts[trk] = 0
                trk.metadata["hits"] += 1
                if trk.metadata["hits"] >= CONFIRMATION_THRESHOLD:
                    trk.metadata["confirmed"] = True
            else: 
                missed_counts[trk] += 1
            

    for trk in list(tracks):
        if missed_counts[trk] > max_missed:
            tracks.remove(trk)
            

    new_trks = spawn_tracks(meas_set, tracks, now)
    for t in new_trks:
        tracks.add(t)
        all_tracks.add(t)
        missed_counts[t] = 0

print("Simülasyon Tamamlandı.")
print("-" * 40)
print(f"TOPLAM AÇILAN TRACK PENCERESİ SAYISI: {total_track_windows}")
print("-" * 40)


confirmed_tracks_for_plot = {t for t in all_tracks if t.metadata.get("confirmed")}
print(f"Toplam Onaylı İz Sayısı: {len(confirmed_tracks_for_plot)}")

plotter = AnimatedPlotterly(timesteps, tail_length=1)

plotter.plot_ground_truths(truths, [0, 2])
plotter.plot_measurements(all_measurements, [0, 2])
plotter.plot_tracks(confirmed_tracks_for_plot, [0, 2], uncertainty=True)

original_frames = list(plotter.fig.frames)
plotter.fig.frames = original_frames * 3


plotter.fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(
                    label="Play (2 loops)",
                    method="animate",
                    args=[
                        None,
                        {
                            "frame": {"duration": 100, "redraw": True},
                            "transition": {"duration": 0},
                            "fromcurrent": False
                        }
                    ],
                )
            ],
        )
    ]
)

plotter.fig.show()
