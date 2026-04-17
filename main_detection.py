import cv2
import time
import json
import numpy as np
import gridfs
from collections import deque, defaultdict
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO
from pymongo import MongoClient

from parking_violation_detection.tracker import update_tracker
from parking_violation_detection.utils import get_bottom_center
from speed_violation_detection.speed_estimator import SpeedEstimator


# =========================
# CONFIG
# =========================

with open("config/parking_config.json") as f:
    parking_config = json.load(f)

with open("config/speed_config.json") as f:
    speed_config = json.load(f)


# =========================
# STORAGE SETUP (SSD + MONGO + GRIDFS)
# =========================

MONGO_URI = "mongodb+srv://zebra_backend:backend_zebra@smart-zebra-crossing-cl.kahl9t8.mongodb.net/violation_db?retryWrites=true&w=majority&appName=smart-zebra-crossing-cluster"

client = MongoClient(MONGO_URI)
db = client["violation_db"]
collection = db["violations"]
fs = gridfs.GridFS(db)

ssd_root = Path("violations_storage")
video_dir = ssd_root / "videos"

video_dir.mkdir(parents=True, exist_ok=True)


# =========================
# MODEL
# =========================

model = YOLO(speed_config["model_path"])


# =========================
# HELPERS
# =========================

def inside_polygon(x, y, poly):
    return cv2.pointPolygonTest(poly, (x, y), False) >= 0


def is_stationary(positions, threshold=5):
    if len(positions) < 5:
        return False

    d = [
        np.linalg.norm(np.array(positions[i]) - np.array(positions[i - 1]))
        for i in range(1, len(positions))
    ]
    return np.mean(d) < threshold


# =========================
# ENGINE
# =========================

class ViolationEngine:
    def __init__(self, video_path, fps=30):

        self.cap = cv2.VideoCapture(video_path)
        self.fps = fps

        self.model = model
        self.estimator = SpeedEstimator(fps, speed_config["distance_meters"])

        self.frame_buffer = deque(maxlen=int(fps * 5))

        self.vehicle_positions = defaultdict(lambda: deque(maxlen=10))
        self.vehicle_entry_time = {}

        self.speed_buffer = defaultdict(int)
        self.park_buffer = defaultdict(int)

        self.speed_violations = set()
        self.parking_violations = set()
        self.logged = set()

        self.zone1 = tuple(map(tuple, speed_config["line1"]))
        self.zone2 = tuple(map(tuple, speed_config["line2"]))

        self.zebra_zone = np.array(parking_config["zebra_zone"], dtype=np.int32)
        self.buffer_zone = np.array(parking_config["buffer_zone"], dtype=np.int32)

    # =========================
    # SAVE FUNCTION (SSD + MONGO + GRIDFS)
    # =========================

    def save_violation(self, frame, tid, vtype, speed):

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        video_path = video_dir / f"{vtype}_{tid}_{ts}.mp4"
        image_filename = f"{vtype}_{tid}_{ts}.jpg"

        h, w = frame.shape[:2]

        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (w, h)
        )

        for f in self.frame_buffer:
            writer.write(f)
        writer.release()

        success, encoded_img = cv2.imencode('.jpg', frame)
        if not success:
            raise RuntimeError(f"Failed to encode image for violation {tid}")
        image_bytes = encoded_img.tobytes()

        # GRIDFS
        grid_id = fs.put(
            image_bytes,
            filename=image_filename,
            track_id=tid,
            type=vtype,
            speed=speed,
            time=datetime.now()
        )

        # MONGO METADATA
        collection.insert_one({
            "track_id": tid,
            "type": vtype,
            "speed": speed,
            "time": datetime.now(),
            "video_path": str(video_path),
            "image_filename": image_filename,
            "gridfs_image_id": grid_id,
            "source": "violation_engine_v1"
        })

    # =========================
    # MAIN LOOP
    # =========================

    def run(self):

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_buffer.append(frame.copy())

            frame_id = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

            results = self.model(frame, conf=0.25, verbose=False)[0]

            detections = []
            for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf.item(), int(cls)))

            tracks = update_tracker(detections, frame)

            for tid, x1, y1, x2, y2, cls in tracks:

                cx, cy = get_bottom_center(x1, y1, x2, y2)

                # =========================
                # SPEED
                # =========================
                measurement, _ = self.estimator.update(
                    tid, (cx, cy), frame_id, self.zone1, self.zone2
                )

                speed = measurement["speed_kmph"] if measurement else 0

                if speed > speed_config["speed_limit_kmph"]:
                    self.speed_buffer[tid] += 1
                else:
                    self.speed_buffer[tid] = max(0, self.speed_buffer[tid] - 1)

                if self.speed_buffer[tid] >= 5:
                    self.speed_violations.add(tid)

                # =========================
                # PARKING
                # =========================

                self.vehicle_positions[tid].append((cx, cy))

                inside = (
                    inside_polygon(cx, cy, self.zebra_zone) or
                    inside_polygon(cx, cy, self.buffer_zone)
                )

                stationary = is_stationary(self.vehicle_positions[tid])

                if stationary and inside:
                    self.park_buffer[tid] += 1
                else:
                    self.park_buffer[tid] = max(0, self.park_buffer[tid] - 1)

                if self.park_buffer[tid] >= int(self.fps * 2):
                    if tid not in self.vehicle_entry_time:
                        self.vehicle_entry_time[tid] = time.time()

                    duration = time.time() - self.vehicle_entry_time[tid]

                    if duration > parking_config["parking_threshold"]:
                        self.parking_violations.add(tid)
                else:
                    self.vehicle_entry_time.pop(tid, None)

                # =========================
                # VIOLATION TYPE
                # =========================

                vtype = None

                if tid in self.speed_violations:
                    vtype = "speed"

                if tid in self.parking_violations:
                    vtype = "parking"

                if tid in self.speed_violations and tid in self.parking_violations:
                    vtype = "both"

                # =========================
                # SAVE + EVENT OUTPUT
                # =========================

                if vtype and tid not in self.logged:
                    self.logged.add(tid)

                    # SAVE EVERYTHING (SSD + MONGO + GRIDFS)
                    self.save_violation(frame, tid, vtype, speed)

                    # EVENT (for future LED/web integration)
                    print({
                        "track_id": tid,
                        "type": vtype,
                        "speed": speed,
                        "time": datetime.now()
                    })

                # =========================
                # DEBUG DISPLAY
                # =========================

                color = (0, 255, 0)
                label = f"{speed:.1f} km/h"

                if tid in self.speed_violations:
                    color = (0, 0, 255)
                    label = "Speed Violation"

                if tid in self.parking_violations:
                    color = (255, 0, 0)
                    label = "Parking Violation"

                if tid in self.speed_violations and tid in self.parking_violations:
                    color = (0, 0, 0)
                    label = "Both Violations"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Violation Engine", frame)

            if cv2.waitKey(1) == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    video_path = speed_config["video_path"]
    engine = ViolationEngine(video_path)
    engine.run()