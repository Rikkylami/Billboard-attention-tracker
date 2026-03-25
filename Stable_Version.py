import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
import sys
import signal
import atexit
from typing import Dict, Optional

sys.stdout = open("tracker_log.txt", "w")
sys.stderr = open("tracker_errors.txt", "w")

def _flush_logs(*args: object, **kwargs: object) -> None:
    sys.stdout.flush()
    sys.stderr.flush()

atexit.register(_flush_logs)

# Classes 

class IDGenerator:
    def __init__(self):
        self._id = 0

    def get_next(self) -> int:
        self._id += 1
        return self._id

class Stabilizer:
    def __init__(self, smoothing: float = 0.85):
        self.state: Optional[float] = None
        self.smoothing = smoothing

    def update(self, measurement: float) -> float:
        state = self.state
        if state is None:
            new_state = measurement
        else:
            new_state = (self.smoothing * state) + ((1 - self.smoothing) * measurement)
        self.state = new_state
        return new_state

class Viewer:
    def __init__(self, centroid):
        self.centroid = centroid
        self.x_stab = Stabilizer()
        self.y_stab = Stabilizer()
        self.is_looking = False
        self.look_start_time = 0.0
        self.last_look_time = 0.0
        self.last_seen_time = time.time()
        self.look_away_start: Optional[float] = None

# Constants 

AUTO_CALIB_SECONDS    = 6.0
CALIB_PADDING_FRAC    = 0.15
CALIB_FALLBACK_HALF_X = 220.0
CALIB_FALLBACK_HALF_Y = 140.0

SCREEN_WIDTH           = 1920
SCREEN_HEIGHT          = 1080
GAZE_THRESHOLD_SECONDS = 4.0
GRACE_PERIOD_SECONDS   = 1.5
CSV_FILENAME           = "billboard_analytics.csv"

# Calibration State 

_calib_locked = False
_calib_samples_x: list = []
_calib_samples_y: list = []
_calib_start: Optional[float] = None

_calib_min_x = -CALIB_FALLBACK_HALF_X
_calib_max_x =  CALIB_FALLBACK_HALF_X
_calib_min_y = -CALIB_FALLBACK_HALF_Y
_calib_max_y =  CALIB_FALLBACK_HALF_Y

def _update_calibration(raw_x: float, raw_y: float) -> bool:
    global _calib_locked, _calib_start
    global _calib_min_x, _calib_max_x, _calib_min_y, _calib_max_y

    if _calib_locked:
        return True

    now = time.time()
    if _calib_start is None:
        _calib_start = now

    _calib_samples_x.append(raw_x)
    _calib_samples_y.append(raw_y)

    if (now - _calib_start) >= AUTO_CALIB_SECONDS and len(_calib_samples_x) >= 10:
        obs_min_x, obs_max_x = min(_calib_samples_x), max(_calib_samples_x)
        obs_min_y, obs_max_y = min(_calib_samples_y), max(_calib_samples_y)

        range_x = max(obs_max_x - obs_min_x, 1.0)
        range_y = max(obs_max_y - obs_min_y, 1.0)

        if range_x < 50:
            print("[Warning] X range small results may be unreliable", flush=True)
        if range_y < 30:
            print("[Warning] Y range small results may be unreliable", flush=True)

        pad_x = range_x * CALIB_PADDING_FRAC
        pad_y = range_y * CALIB_PADDING_FRAC

        _calib_min_x = obs_min_x - pad_x
        _calib_max_x = obs_max_x + pad_x
        _calib_min_y = obs_min_y - pad_y
        _calib_max_y = obs_max_y + pad_y

        _calib_locked = True
        print(f"[Locked] X {int(_calib_min_x)} {int(_calib_max_x)} Y {int(_calib_min_y)} {int(_calib_max_y)}", flush=True)

    return _calib_locked

# MediaPipe Setup 

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=10,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

FACE_2D_INDICES = [33, 263, 1, 61, 291, 199]
FACE_3D_MODEL = np.array([
    [-225.0,  170.0, -135.0],
    [ 225.0,  170.0, -135.0],
    [   0.0,    0.0,    0.0],
    [-150.0, -150.0, -125.0],
    [ 150.0, -150.0, -125.0],
    [   0.0, -330.0,  -65.0],
], dtype=np.float64)

# Analytics  

total_views_state = {"count": 0}
active_viewers: Dict[int, Viewer] = {}
id_gen = IDGenerator()

if not os.path.isfile(CSV_FILENAME):
    with open(CSV_FILENAME, mode="a", newline="") as file:
        csv.writer(file).writerow(["Timestamp", "Duration_Seconds", "View_Number"])

def log_view(duration: float) -> None:
    total_views_state["count"] += 1
    timestamp = time.strftime("%Y%m%d %H:%M:%S")
    with open(CSV_FILENAME, mode="a", newline="") as file:
        csv.writer(file).writerow(
            [timestamp, round(duration, 1), total_views_state["count"]]
        )

# Camera Setup 

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR Camera failed", flush=True)
    sys.exit(1)

img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
focal_length = float(img_w)
cam_matrix = np.array([
    [focal_length, 0,            img_w / 2],
    [0,            focal_length, img_h / 2],
    [0,            0,            1        ],
], dtype=np.float64)
dist_matrix = np.zeros((4, 1), dtype=np.float64)

def _shutdown(sig, frame):
    cap.release()
    sys.exit(0)

signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT, _shutdown)

# Main Loop 

_read_failures = 0
MAX_READ_FAILURES = 30

while cap.isOpened():
    success, image = cap.read()

    if not success:
        _read_failures += 1
        if _read_failures >= MAX_READ_FAILURES:
            break
        time.sleep(0.05)
        continue
    _read_failures = 0

    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_image)
    current_frame_viewers = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_2d = []
            sum_x, sum_y = 0, 0
            for idx in FACE_2D_INDICES:
                pt = face_landmarks.landmark[idx]
                x, y = int(pt.x * img_w), int(pt.y * img_h)
                face_2d.append([x, y])
                sum_x += x
                sum_y += y

            centroid = (sum_x / len(FACE_2D_INDICES), sum_y / len(FACE_2D_INDICES))
            face_2d = np.array(face_2d, dtype=np.float64)
            success_pnp, rvec, tvec = cv2.solvePnP(
                FACE_3D_MODEL, face_2d, cam_matrix, dist_matrix
            )

            if success_pnp:
                rmat, _ = cv2.Rodrigues(rvec)
                forward_vector = rmat[:, 2]
                ray_origin = tvec.flatten()
                plane_normal = np.array([0.0, 0.0, -1.0])
                plane_point  = np.array([0.0, 0.0, 500.0])
                denominator  = np.dot(forward_vector, plane_normal)

                if abs(denominator) > 1e-6:
                    t = np.dot(plane_point - ray_origin, plane_normal) / denominator
                    if t > 0:
                        intersection_3d = ray_origin + (forward_vector * t)
                        raw_x = intersection_3d[0]
                        raw_y = intersection_3d[1]

                        _update_calibration(raw_x, raw_y)
                        current_frame_viewers.append((centroid, raw_x, raw_y))

    matched_ids  = set()
    current_time = time.time()

    for centroid, raw_x, raw_y in current_frame_viewers:
        best_id     = None
        min_dist    = float("inf")
        best_viewer = None

        for vid, viewer in active_viewers.items():
            if viewer is None:
                continue
            if vid in matched_ids:
                continue
            dist = np.hypot(
                centroid[0] - viewer.centroid[0],
                centroid[1] - viewer.centroid[1],
            )
            if dist < 150 and dist < min_dist:
                min_dist    = dist
                best_id     = vid
                best_viewer = viewer

        if best_viewer is None:
            best_id     = id_gen.get_next()
            best_viewer = Viewer(centroid)
            active_viewers[best_id] = best_viewer
            
        assert best_id is not None
        matched_ids.add(best_id)
        
        viewer = best_viewer
        assert viewer is not None
        viewer.centroid       = centroid
        viewer.last_seen_time = current_time

        smooth_x = viewer.x_stab.update(float(raw_x))
        smooth_y = viewer.y_stab.update(float(raw_y))

        pixel_x = np.interp(smooth_x, [_calib_min_x, _calib_max_x], [0, SCREEN_WIDTH])
        pixel_y = np.interp(smooth_y, [_calib_min_y, _calib_max_y], [0, SCREEN_HEIGHT])

        on_billboard = (
            (-50 <= pixel_x <= SCREEN_WIDTH  + 50) and
            (-50 <= pixel_y <= SCREEN_HEIGHT + 50)
        )

        if on_billboard:
            look_away_start = viewer.look_away_start
            if not viewer.is_looking:
                viewer.is_looking      = True
                viewer.look_start_time = current_time
            elif look_away_start is not None:
                away_gap = current_time - look_away_start
                viewer.look_start_time += away_gap
                viewer.look_away_start  = None
            viewer.last_look_time = current_time
        else:
            if viewer.is_looking:
                look_away_start = viewer.look_away_start
                if look_away_start is None:
                    viewer.look_away_start = current_time
                elif (current_time - look_away_start) > GRACE_PERIOD_SECONDS:
                    final_duration = viewer.last_look_time - viewer.look_start_time
                    if final_duration >= GAZE_THRESHOLD_SECONDS:
                        log_view(final_duration)
                    viewer.is_looking      = False
                    viewer.look_away_start = None

    for vid, viewer in list(active_viewers.items()):
        if vid not in matched_ids:
            if (current_time - viewer.last_seen_time) > GRACE_PERIOD_SECONDS:
                if viewer.is_looking:
                    final_duration = viewer.last_look_time - viewer.look_start_time
                    if final_duration >= GAZE_THRESHOLD_SECONDS:
                        log_view(final_duration)
                active_viewers.pop(vid, None)

    if not _calib_locked:
        calib_start_val = _calib_start
        if calib_start_val is not None:
            time_left = max(0.0, AUTO_CALIB_SECONDS - (current_time - calib_start_val))
            cv2.putText(image, f"CALIBRATING {time_left:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    else:
        cv2.putText(image, f"ACTIVE Views {total_views_state['count']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # cv2.imshow('DOOH Tracker', image)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
#cv2.destroyAllWindows()