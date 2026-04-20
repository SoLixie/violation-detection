#!/usr/bin/env python3

import cv2
import argparse
import json
import numpy as np
import time
import sys
import logging
from pathlib import Path
from collections import deque, defaultdict

from ultralytics import YOLO

# =========================================================
# PROJECT ROOT
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SmartZebra")

# =========================================================
# INTERNAL IMPORTS
# =========================================================
from tracker import update_tracker
from common.geometry import (
    get_bottom_center,
    is_inside_polygon,
    is_stationary
)

from speed_violation_detection.speed_estimator import SpeedEstimator

from visual_utils import (
    draw_speed_zones,
    draw_parking_zones,
    draw_status_hud,
    draw_vehicle_label,
    setup_display_window,
    get_color
)

from storage.mongo_handler import get_mongo_handler

# =========================================================
# CONFIG
# =========================================================
VEHICLE_CLASSES = {1, 2, 3, 5, 7}

def load_config(path):
    config_path = PROJECT_ROOT / path
    if not config_path.exists():
        raise FileNotFoundError(path)
    with open(config_path) as f:
        return json.load(f)

speed_config = load_config("config/speed_config.json")
parking_config = load_config("config/parking_config.json")

# =========================================================
# ARGS
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", type=int, default=None)
    parser.add_argument("--video", type=str, default=None)
    return parser.parse_args()

# =========================================================
# MAIN
# =========================================================
def main():

    args = parse_args()

    # ---------------- VIDEO ----------------
    if args.live is not None:
        cap = cv2.VideoCapture(args.live)
    elif args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(str(PROJECT_ROOT / speed_config["video_path"]))

    if not cap.isOpened():
        raise RuntimeError("Cannot open video source")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # ---------------- MODEL ----------------
    model = YOLO(str(PROJECT_ROOT / "best.pt"))

    # ---------------- SPEED ----------------
    speed_estimator = SpeedEstimator(
        fps,
        speed_config["distance_meters"]
    )

    # ---------------- MONGO ----------------
    mongo = get_mongo_handler()

    # ---------------- SSD STORAGE ----------------
    ssd_dir = Path("violations_storage/videos")
    ssd_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # STATE
    # =========================================================
    vehicle_positions = defaultdict(lambda: deque(maxlen=30))
    vehicle_entry_time = {}
    last_speed = defaultdict(float)

    locked_state = {}
    logged = set()

    violation_buffer = defaultdict(lambda: deque(maxlen=30))
    recording_active = {}

    total_violations = 0
    speed_violation_count = 0
    parking_violation_count = 0

    violation_triggered = defaultdict(lambda: {
        "speed": False,
        "parking": False
    })

    # ---------------- ZONES ----------------
    speed_line1 = np.array(speed_config["line1"], np.int32)
    speed_line2 = np.array(speed_config["line2"], np.int32)

    parking_zone = np.array(parking_config["zebra_zone"], np.int32)
    buffer_zone = np.array(parking_config.get("buffer_zone", []), np.int32)

    setup_display_window("Smart Zebra", 1280, 720)

    frame_idx = 0

    # =========================================================
    while True:

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        annotated = frame.copy()

        results = model(frame, conf=0.25, verbose=False)[0]

        detections = []

        for box, conf, cls in zip(results.boxes.xyxy,
                                   results.boxes.conf,
                                   results.boxes.cls):

            if int(cls) in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box)
                detections.append(([x1, y1, x2 - x1, y2 - y1], float(conf), int(cls)))

        tracks = update_tracker(detections, frame)

        active = set()

        # =========================================================
        # VEHICLE LOOP
        # =========================================================
        for tid, x1, y1, x2, y2, cls in tracks:

            active.add(tid)

            cx, cy = get_bottom_center(x1, y1, x2, y2)
            vehicle_positions[tid].append((cx, cy))

            violation_buffer[tid].append(frame.copy())

            # ---------------- SPEED ----------------
            measurement = speed_estimator.update(
                tid,
                (cx, cy),
                frame_idx,
                speed_line1.tolist(),
                speed_line2.tolist()
            )

            speed = last_speed.get(tid, 0.0)

            if measurement and "speed_kmph" in measurement:
                speed = float(measurement["speed_kmph"])
                last_speed[tid] = speed

            speed_violation = speed > speed_config["speed_limit_kmph"]

            # ---------------- PARKING ----------------
            stationary = is_stationary(vehicle_positions[tid])

            in_zone = (
                is_inside_polygon(cx, cy, parking_zone) or
                (len(buffer_zone) >= 3 and is_inside_polygon(cx, cy, buffer_zone))
            )

            parking_violation = False

            if stationary and in_zone:
                if tid not in vehicle_entry_time:
                    vehicle_entry_time[tid] = time.time()

                duration = time.time() - vehicle_entry_time[tid]
                parking_violation = duration > parking_config.get("parking_threshold", 10)
            else:
                vehicle_entry_time.pop(tid, None)

            # ---------------- STATE ----------------
            prev = locked_state.get(tid, "normal")

            if speed_violation and parking_violation:
                vtype = "both"
            elif speed_violation:
                vtype = "speed"
            elif parking_violation:
                vtype = "parking"
            elif prev in ("speed", "parking", "both"):
                vtype = prev
            else:
                vtype = "normal"

            locked_state[tid] = vtype

            # =========================================================
            # COUNTERS (NO SPAM)
            # =========================================================
            if speed_violation:
                if not violation_triggered[tid]["speed"]:
                    speed_violation_count += 1
                    violation_triggered[tid]["speed"] = True
            else:
                violation_triggered[tid]["speed"] = False

            if parking_violation:
                if not violation_triggered[tid]["parking"]:
                    parking_violation_count += 1
                    violation_triggered[tid]["parking"] = True
            else:
                violation_triggered[tid]["parking"] = False

            # =========================================================
            # EVENT START
            # =========================================================
            if vtype in ("speed", "parking", "both"):

                if (tid, vtype) not in logged:
                    logged.add((tid, vtype))
                    total_violations += 1

                    logger.info(f"🚨 SPEED | ID={tid} | Speed={int(speed)} km/h")

                    # ---------------- MONGO IMAGE ----------------
                    if mongo:
                        try:
                            mongo.save_violation(
                                annotated,
                                tid,
                                vtype,
                                float(speed),
                                metadata={
                                    "type": vtype,
                                    "speed": float(speed),
                                    "timestamp": time.time()
                                }
                            )
                        except Exception as e:
                            logger.warning(f"Mongo failed: {e}")

                    # ---------------- START VIDEO CLIP ----------------
                    h, w = frame.shape[:2]
                    path = ssd_dir / f"violation_{tid}_{int(time.time())}.mp4"

                    writer = cv2.VideoWriter(
                        str(path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (w, h)
                    )

                    # write buffered frames
                    for f in violation_buffer[tid]:
                        writer.write(f)

                    recording_active[tid] = writer

            # =========================================================
            # CONTINUE RECORDING
            # =========================================================
            if tid in recording_active:
                recording_active[tid].write(frame)

            # =========================================================
            # STOP RECORDING
            # =========================================================
            if vtype == "normal" and tid in recording_active:
                recording_active[tid].release()
                del recording_active[tid]
                violation_buffer[tid].clear()

            # ---------------- DRAW ----------------
            has_speed = speed > 0

            draw_vehicle_label(
                annotated,
                (x1, y1, x2, y2),
                tid,
                speed,
                has_speed,
                vtype
            )

        # =========================================================
        # CLEANUP
        # =========================================================
        for tid in list(vehicle_positions.keys()):
            if tid not in active:
                vehicle_positions.pop(tid, None)
                vehicle_entry_time.pop(tid, None)

        # =========================================================
        # UI
        # =========================================================
        draw_speed_zones(annotated, speed_line1, speed_line2)
        draw_parking_zones(annotated, parking_zone, buffer_zone)

        draw_status_hud(
            annotated,
            total_violations,
            f"Speed Limit {speed_config['speed_limit_kmph']} km/h",
            speed_violation_count,
            parking_violation_count
        )

        cv2.imshow("Smart Zebra", annotated)

        frame_idx += 1

        if cv2.waitKey(1) == 27:
            break

    cap.release()

    for w in recording_active.values():
        w.release()

    cv2.destroyAllWindows()


# =========================================================
if __name__ == "__main__":
    main()