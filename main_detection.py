import cv2
import time
import json
import queue
import threading
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
# MODEL LOADER
# =========================

def load_detection_model(model_path, use_tpu=False):
    ext = Path(model_path).suffix.lower()
    if ext == ".tflite":
        if Interpreter is None:
            raise RuntimeError(
                "tflite-runtime or TensorFlow is required to run .tflite models"
            )

        interpreter_args = {"model_path": str(model_path)}
        if use_tpu:
            if load_delegate is None:
                raise RuntimeError(
                    "Edge TPU delegate is required for Coral TPU support"
                )
            interpreter_args["experimental_delegates"] = [
                load_delegate("libedgetpu.so.1")
            ]

        interpreter = Interpreter(**interpreter_args)
        interpreter.allocate_tensors()
        return {
            "type": "tflite",
            "interpreter": interpreter,
            "input_details": interpreter.get_input_details(),
            "output_details": interpreter.get_output_details(),
        }

    return {"type": "yolo", "model": YOLO(model_path)}


class DetectionBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls


class DetectionResult:
    def __init__(self, boxes):
        self.boxes = boxes


def decode_tflite_detections(model, frame, conf_threshold=0.25):
    interpreter = model["interpreter"]
    input_details = model["input_details"]
    output_details = model["output_details"]

    input_shape = input_details[0]["shape"]
    input_height, input_width = int(input_shape[1]), int(input_shape[2])

    resized = cv2.resize(frame, (input_width, input_height))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(rgb, axis=0)

    if input_details[0]["dtype"] == np.float32:
        input_data = input_data.astype(np.float32) / 255.0
    else:
        input_data = input_data.astype(input_details[0]["dtype"])

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    outputs = [interpreter.get_tensor(o["index"]) for o in output_details]

    boxes = np.empty((0, 4), dtype=np.float32)
    scores = np.empty((0,), dtype=np.float32)
    classes = np.empty((0,), dtype=np.int32)

    if len(outputs) >= 1 and outputs[0].ndim == 3 and outputs[0].shape[-1] > 5:
        data = outputs[0][0]
        xywh = data[:, :4]
        obj_conf = data[:, 4]
        class_probs = data[:, 5:]
        class_ids = np.argmax(class_probs, axis=-1).astype(np.int32)
        class_scores = class_probs[np.arange(len(class_ids)), class_ids]
        confs = obj_conf * class_scores

        mask = confs >= conf_threshold
        xywh = xywh[mask]
        confs = confs[mask]
        class_ids = class_ids[mask]

        if xywh.shape[0] > 0:
            xyxy = np.zeros((xywh.shape[0], 4), dtype=np.float32)
            xyxy[:, 0] = xywh[:, 0] - (xywh[:, 2] / 2)
            xyxy[:, 1] = xywh[:, 1] - (xywh[:, 3] / 2)
            xyxy[:, 2] = xywh[:, 0] + (xywh[:, 2] / 2)
            xyxy[:, 3] = xywh[:, 1] + (xywh[:, 3] / 2)

            scale_x = frame.shape[1] / input_width
            scale_y = frame.shape[0] / input_height
            xyxy[:, [0, 2]] *= scale_x
            xyxy[:, [1, 3]] *= scale_y

            rects = xywh.copy()
            rects[:, 0] = rects[:, 0] - rects[:, 2] / 2
            rects[:, 1] = rects[:, 1] - rects[:, 3] / 2
            rects[:, 2] = rects[:, 2]
            rects[:, 3] = rects[:, 3]
            rects[:, [0, 2]] *= scale_x
            rects[:, [1, 3]] *= scale_y

            indices = cv2.dnn.NMSBoxes(
                rects.tolist(),
                confs.tolist(),
                conf_threshold,
                0.45,
            )
            if len(indices) > 0:
                indices = np.array(indices).flatten()
                boxes = xyxy[indices]
                scores = confs[indices]
                classes = class_ids[indices]
    elif len(outputs) >= 3:
        raw_boxes = outputs[0][0]
        raw_scores = outputs[1][0]
        raw_classes = outputs[2][0].astype(np.int32)
        mask = raw_scores >= conf_threshold
        if np.any(mask):
            boxes = raw_boxes[mask]
            scores = raw_scores[mask]
            classes = raw_classes[mask]

            scale_x = frame.shape[1] / input_width
            scale_y = frame.shape[0] / input_height
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

    return DetectionResult(DetectionBoxes(boxes, scores, classes))


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
# SAVE WORKER
# =========================

class ViolationSaver(threading.Thread):
    def __init__(self, collection, fs):
        super().__init__(daemon=True)
        self.collection = collection
        self.fs = fs
        self.tasks = queue.Queue()
        self._stopped = threading.Event()
        self.start()

    def enqueue(self, video_frames, frame, tid, vtype, speed):
        self.tasks.put((video_frames, frame, tid, vtype, speed))

    def run(self):
        while not self._stopped.is_set() or not self.tasks.empty():
            try:
                video_frames, frame, tid, vtype, speed = self.tasks.get(timeout=0.25)
            except queue.Empty:
                continue

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = video_dir / f"{vtype}_{tid}_{ts}.mp4"
            image_filename = f"{vtype}_{tid}_{ts}.jpg"

            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                30,
                (w, h)
            )
            for f in video_frames:
                writer.write(f)
            writer.release()

            success, encoded_img = cv2.imencode('.jpg', frame)
            if not success:
                self.tasks.task_done()
                continue
            image_bytes = encoded_img.tobytes()

            grid_id = self.fs.put(
                image_bytes,
                filename=image_filename,
                track_id=tid,
                type=vtype,
                speed=speed,
                time=datetime.now()
            )

            self.collection.insert_one({
                "track_id": tid,
                "type": vtype,
                "speed": speed,
                "time": datetime.now(),
                "video_path": str(video_path),
                "image_filename": image_filename,
                "gridfs_image_id": grid_id,
                "source": "violation_engine_v1"
            })
            self.tasks.task_done()

    def stop(self):
        self._stopped.set()
        self.join(timeout=5)


# =========================
# ENGINE
# =========================

class ViolationEngine:
    def __init__(self, video_path, fps=30):

        self.cap = cv2.VideoCapture(video_path)
        self.fps = fps

        loaded_model = load_detection_model(
            speed_config["model_path"],
            speed_config.get("use_tpu", False),
        )
        self.model_type = loaded_model["type"]
        self.model = loaded_model

        self.estimator = SpeedEstimator(fps, speed_config["distance_meters"])
        self.imgsz = speed_config.get("imgsz", 640)
        self.display = speed_config.get("display_window", False)

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

        self.saver = ViolationSaver(collection, fs)

    # =========================
    # SAVE FUNCTION (SSD + MONGO + GRIDFS)
    # =========================

    def save_violation(self, frame, tid, vtype, speed):
        self.saver.enqueue(list(self.frame_buffer), frame.copy(), tid, vtype, speed)

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

            if self.model_type == "yolo":
                results = self.model["model"](
                    frame, conf=0.25, imgsz=self.imgsz, verbose=False
                )[0]
            else:
                results = decode_tflite_detections(self.model, frame, conf_threshold=0.25)

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

            if self.display:
                cv2.imshow("Violation Engine", frame)
                if cv2.waitKey(1) == 27:
                    break

        self.saver.stop()
        self.cap.release()
        if self.display:
            cv2.destroyAllWindows()


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    video_path = speed_config["video_path"]
    engine = ViolationEngine(video_path)
    engine.run()