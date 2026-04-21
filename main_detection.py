#!/usr/bin/env python3

import cv2
import argparse
import json
import numpy as np
import time
import sys
import logging
import importlib
import os
from pathlib import Path
from collections import deque, defaultdict

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    _tflite_module = importlib.import_module("tflite_runtime.interpreter")
    Interpreter = _tflite_module.Interpreter
    load_delegate = _tflite_module.load_delegate
except ImportError:
    try:
        _tf_lite_module = importlib.import_module("tensorflow.lite.python.interpreter")
        Interpreter = _tf_lite_module.Interpreter
        load_delegate = _tf_lite_module.load_delegate
    except ImportError:
        Interpreter = None
        load_delegate = None

# =========================================================
# PROJECT ROOT
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SmartZebra")

# =========================================================
# INTERNAL IMPORTS (FIXED PATHS)
# =========================================================
from core.tracker import update_tracker
from core.geometry import (
    get_bottom_center,
    is_inside_polygon,
    is_stationary
)

from engine.speed_estimator import SpeedEstimator

from core.visual_utils import (
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

# FIXED: single config file
config = load_config("config/smart_config.json")

# ---------------- SPEED ----------------
speed_config = {
    "video_path": config["source"],
    "distance_meters": float(config.get("distance_meters", 10.0)),
    "speed_limit_kmph": float(config.get("speed_limit_kmph", 50.0)),
    "line1": config["speed_lines"]["line1"],
    "line2": config["speed_lines"]["line2"]
}

# ---------------- PARKING ----------------
parking_config = {
    "zebra_zone": config["parking_zones"]["zebra_zone"],
    "buffer_zone": config["parking_zones"]["buffer_zone"],
    "parking_threshold": float(config.get("parking_threshold", 10.0))
}

# =========================================================
# ARGS
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", type=int, default=None)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--model", type=str, default=None, help="Path to a .pt or .tflite model.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for saved violation clips. Defaults to project_root/violations_storage/videos."
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size for PT models.")
    parser.add_argument(
        "--max-width",
        type=int,
        default=1280,
        help="Resize video frames to this width before detection, tracking, and recording. Use 0 to disable."
    )
    parser.add_argument(
        "--buffer-frames",
        type=int,
        default=12,
        help="Number of pre-event frames to keep for saved violation clips."
    )
    parser.add_argument(
        "--no-realtime-sync",
        action="store_true",
        help="Disable real-time sync for video files and process every frame sequentially."
    )
    parser.add_argument(
        "--allow-frame-skip-for-speed",
        action="store_true",
        help="Allow real-time sync to skip frames even when speed detection is enabled."
    )
    parser.add_argument(
        "--tpu",
        action="store_true",
        help="Use the Edge TPU delegate for TFLite models."
    )
    parser.add_argument(
        "--config",
        choices=("speed", "parking", "both"),
        default="both",
        help="Enable speed detection, parking detection, or both."
    )
    return parser.parse_args()


def resolve_video_source(args):
    if args.live is not None:
        return args.live, True

    candidate = Path(args.video) if args.video else Path(speed_config["video_path"])
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()

    if not candidate.exists():
        raise FileNotFoundError(f"Video source not found: {candidate}")

    return str(candidate), False


def resolve_model_path(args):
    if args.model:
        raw_model_path = Path(args.model)
        candidates = []

        if raw_model_path.is_absolute():
            candidates.append(raw_model_path)
        else:
            candidates.append(Path.cwd() / raw_model_path)
            candidates.append(PROJECT_ROOT / raw_model_path)

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.exists():
                return resolved

        raise FileNotFoundError(f"Model not found: {candidates[0].resolve()}")

    default_candidates = [
        PROJECT_ROOT / "model" / "best.pt",
        PROJECT_ROOT / "model" / "best.tflite",
        PROJECT_ROOT / "model" / "best_edgetpu.tflite",
    ]
    for candidate in default_candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"No default model found in {PROJECT_ROOT / 'model'}. "
        "Expected best.pt, best.tflite, or best_edgetpu.tflite."
    )


def cleanup_track(
    tid,
    vehicle_positions,
    vehicle_entry_time,
    last_speed,
    locked_state,
    recording_active,
    violation_triggered,
    speed_estimator
):
    vehicle_positions.pop(tid, None)
    vehicle_entry_time.pop(tid, None)
    last_speed.pop(tid, None)
    locked_state.pop(tid, None)
    violation_triggered.pop(tid, None)
    speed_estimator.reset_track(tid)

    writer = recording_active.pop(tid, None)
    if writer is not None:
        writer.release()


def resolve_output_dir(args):
    configured_output_dir = args.output_dir or os.getenv("SMART_ZEBRA_OUTPUT_DIR", "").strip()
    if configured_output_dir:
        output_dir = Path(configured_output_dir)
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir
    else:
        output_dir = PROJECT_ROOT / "violations_storage" / "videos"

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_processing_size(source_width, source_height, max_width):
    if max_width <= 0 or source_width <= max_width:
        return int(source_width), int(source_height)

    scale = max_width / float(source_width)
    return int(max_width), max(1, int(round(source_height * scale)))


def resize_frame_for_processing(frame, target_width, target_height):
    current_height, current_width = frame.shape[:2]
    if current_width == target_width and current_height == target_height:
        return frame
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def scale_points(points, scale_x, scale_y):
    points_array = np.array(points, dtype=np.float32)
    if points_array.size == 0:
        return np.array(points, np.int32)

    points_array[..., 0] *= scale_x
    points_array[..., 1] *= scale_y
    return np.rint(points_array).astype(np.int32)


def sync_video_to_realtime(cap, fps, frame_idx, start_time, enabled):
    if not enabled or fps <= 0:
        return frame_idx

    target_frame_idx = int((time.perf_counter() - start_time) * fps)
    frames_to_skip = target_frame_idx - frame_idx - 1

    if frames_to_skip <= 0:
        return frame_idx

    skipped = 0
    for _ in range(frames_to_skip):
        if not cap.grab():
            break
        skipped += 1

    if skipped > 0:
        logger.info("Realtime sync skipped %s frame(s)", skipped)

    return frame_idx + skipped


def get_frame_timestamp_seconds(cap, is_live_source, frame_idx, fps, live_start_time):
    if is_live_source:
        return time.perf_counter() - live_start_time

    timestamp_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    if timestamp_msec and timestamp_msec > 0:
        return timestamp_msec / 1000.0

    if fps > 0:
        return frame_idx / fps

    return 0.0


class PTDetector:
    def __init__(self, model_path, conf_threshold=0.25, image_size=640):
        if YOLO is None:
            raise RuntimeError("Ultralytics is not installed. Install it to use .pt models.")
        self.model = YOLO(str(model_path))
        self.conf_threshold = conf_threshold
        self.image_size = image_size

    def infer(self, frame):
        results = self.model(frame, conf=self.conf_threshold, imgsz=self.image_size, verbose=False)[0]
        detections = []

        for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            class_id = int(cls)
            if class_id not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box)
            detections.append(([x1, y1, x2 - x1, y2 - y1], float(conf), class_id))

        return detections


class TFLiteDetector:
    def __init__(self, model_path, conf_threshold=0.25, use_tpu=False):
        if Interpreter is None:
            raise RuntimeError(
                "TFLite runtime is not installed. Install tflite-runtime or tensorflow to use .tflite models."
            )

        delegates = []
        if use_tpu:
            if load_delegate is None:
                raise RuntimeError("Edge TPU delegate loader is unavailable in this Python environment.")

            delegate_names = ["libedgetpu.so.1", "edgetpu.dll", "libedgetpu.1.dylib"]
            delegate_error = None
            for delegate_name in delegate_names:
                try:
                    delegates = [load_delegate(delegate_name)]
                    break
                except Exception as exc:
                    delegate_error = exc

            if not delegates:
                raise RuntimeError(f"Unable to load Edge TPU delegate: {delegate_error}")

        self.interpreter = Interpreter(model_path=str(model_path), experimental_delegates=delegates)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.conf_threshold = conf_threshold

        input_shape = self.input_details[0]["shape"]
        self.input_height = int(input_shape[1])
        self.input_width = int(input_shape[2])
        self.input_dtype = self.input_details[0]["dtype"]
        self.input_quantization = self.input_details[0].get("quantization", (0.0, 0))

    def _set_input(self, frame):
        original_h, original_w = frame.shape[:2]
        resized = cv2.resize(frame, (self.input_width, self.input_height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_tensor = np.expand_dims(rgb, axis=0)

        if np.issubdtype(self.input_dtype, np.floating):
            input_tensor = input_tensor.astype(np.float32) / 255.0
        else:
            scale, zero_point = self.input_quantization
            input_tensor = input_tensor.astype(np.float32)
            if scale and scale > 0:
                input_tensor = np.round(input_tensor / scale + zero_point)
            input_tensor = np.clip(input_tensor, 0, 255).astype(self.input_dtype)

        self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
        return original_w, original_h

    def _get_output_tensors(self):
        tensors = []
        for detail in self.output_details:
            tensor = self.interpreter.get_tensor(detail["index"])
            scale, zero_point = detail.get("quantization", (0.0, 0))
            if not np.issubdtype(tensor.dtype, np.floating) and scale and scale > 0:
                tensor = (tensor.astype(np.float32) - zero_point) * scale
            tensors.append(np.array(tensor))
        return tensors

    def _parse_boxes_with_scores(self, tensor, original_w, original_h):
        detections = []
        array = np.squeeze(tensor)

        if array.ndim == 1:
            array = np.expand_dims(array, axis=0)
        if array.ndim != 2:
            return detections

        if array.shape[-1] < 6:
            return detections

        for row in array:
            x1, y1, x2, y2 = row[:4]
            score = float(row[4])
            class_id = int(row[5])

            if score < self.conf_threshold or class_id not in VEHICLE_CLASSES:
                continue

            if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
                x1 *= original_w
                x2 *= original_w
                y1 *= original_h
                y2 *= original_h

            left = max(0, min(int(x1), original_w - 1))
            top = max(0, min(int(y1), original_h - 1))
            right = max(left + 1, min(int(x2), original_w))
            bottom = max(top + 1, min(int(y2), original_h))
            detections.append(([left, top, right - left, bottom - top], score, class_id))

        return detections

    def _parse_yolo_style(self, tensor, original_w, original_h):
        detections = []
        array = np.squeeze(tensor)

        if array.ndim != 2:
            return detections

        if array.shape[0] > array.shape[1]:
            array = array.T

        if array.shape[1] < 6:
            return detections

        for row in array:
            cx, cy, w, h = row[:4]
            class_scores = row[4:]
            if class_scores.size == 0:
                continue

            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])

            if score < self.conf_threshold or class_id not in VEHICLE_CLASSES:
                continue

            if max(abs(cx), abs(cy), abs(w), abs(h)) <= 2.0:
                cx *= original_w
                w *= original_w
                cy *= original_h
                h *= original_h

            x1 = cx - (w / 2.0)
            y1 = cy - (h / 2.0)
            x2 = cx + (w / 2.0)
            y2 = cy + (h / 2.0)

            left = max(0, min(int(x1), original_w - 1))
            top = max(0, min(int(y1), original_h - 1))
            right = max(left + 1, min(int(x2), original_w))
            bottom = max(top + 1, min(int(y2), original_h))
            detections.append(([left, top, right - left, bottom - top], score, class_id))

        return detections

    def infer(self, frame):
        original_w, original_h = self._set_input(frame)
        self.interpreter.invoke()
        output_tensors = self._get_output_tensors()

        for tensor in output_tensors:
            detections = self._parse_boxes_with_scores(tensor, original_w, original_h)
            if detections:
                return detections

        for tensor in output_tensors:
            detections = self._parse_yolo_style(tensor, original_w, original_h)
            if detections:
                return detections

        return []


def build_detector(model_path, args):
    suffix = model_path.suffix.lower()
    if suffix == ".pt":
        return PTDetector(model_path, conf_threshold=args.conf, image_size=args.imgsz)
    if suffix == ".tflite":
        return TFLiteDetector(model_path, conf_threshold=args.conf, use_tpu=args.tpu)
    raise ValueError(f"Unsupported model format: {model_path.suffix}")

# =========================================================
# MAIN
# =========================================================
def main():

    args = parse_args()
    enable_speed = args.config in ("speed", "both")
    enable_parking = args.config in ("parking", "both")

    # ---------------- VIDEO ----------------
    video_source, is_live_source = resolve_video_source(args)
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {video_source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    if source_width <= 0 or source_height <= 0:
        raise RuntimeError("Unable to read source video dimensions.")

    processing_width, processing_height = get_processing_size(
        source_width,
        source_height,
        args.max_width
    )
    scale_x = processing_width / float(source_width)
    scale_y = processing_height / float(source_height)

    # ---------------- MODEL ----------------
    model_path = resolve_model_path(args)
    detector = build_detector(model_path, args)
    logger.info("Using model backend: %s", model_path.name)
    logger.info(
        "Processing resolution: %sx%s (source %sx%s)",
        processing_width,
        processing_height,
        source_width,
        source_height
    )

    # ---------------- SPEED ----------------
    speed_estimator = SpeedEstimator(
        fps,
        speed_config["distance_meters"]
    )

    # ---------------- MONGO ----------------
    mongo = get_mongo_handler()

    # ---------------- SSD STORAGE ----------------
    ssd_dir = resolve_output_dir(args)

    # =========================================================
    # STATE
    # =========================================================
    vehicle_positions = defaultdict(lambda: deque(maxlen=30))
    vehicle_entry_time = {}
    last_speed = defaultdict(float)

    locked_state = {}
    logged = set()

    pre_event_buffer = deque(maxlen=max(1, args.buffer_frames))
    recording_active = {}

    total_violations = 0
    speed_violation_count = 0
    parking_violation_count = 0

    violation_triggered = defaultdict(lambda: {
        "speed": False,
        "parking": False
    })

    # ---------------- ZONES ----------------
    speed_line1 = scale_points(speed_config["line1"], scale_x, scale_y)
    speed_line2 = scale_points(speed_config["line2"], scale_x, scale_y)

    parking_zone = scale_points(parking_config["zebra_zone"], scale_x, scale_y)
    buffer_zone = scale_points(parking_config.get("buffer_zone", []), scale_x, scale_y)

    setup_display_window("Smart Zebra", 1280, 720)

    frame_idx = 0
    frame_skip_allowed = (not enable_speed) or args.allow_frame_skip_for_speed
    realtime_sync_enabled = (
        (not is_live_source)
        and (not args.no_realtime_sync)
        and fps > 0
        and frame_skip_allowed
    )
    playback_start_time = time.perf_counter()
    live_capture_start_time = time.perf_counter()

    if enable_speed and not frame_skip_allowed:
        logger.info("Frame skipping disabled because speed detection is enabled.")

    # =========================================================
    while True:

        ret, frame = cap.read()
        if not ret:
            if is_live_source:
                logger.warning("Live video source stopped returning frames; ending detection loop.")
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            playback_start_time = time.perf_counter()
            live_capture_start_time = time.perf_counter()
            continue

        frame_timestamp_seconds = get_frame_timestamp_seconds(
            cap,
            is_live_source,
            frame_idx,
            fps,
            live_capture_start_time
        )
        frame = resize_frame_for_processing(frame, processing_width, processing_height)
        annotated = frame.copy()
        pre_event_buffer.append(frame.copy())

        detections = detector.infer(frame)

        tracks = update_tracker(detections, frame)

        active = set()

        # =========================================================
        # VEHICLE LOOP
        # =========================================================
        for tid, x1, y1, x2, y2, cls in tracks:

            active.add(tid)

            cx, cy = get_bottom_center(x1, y1, x2, y2)
            vehicle_positions[tid].append((cx, cy))

            # ---------------- SPEED ----------------
            measurement = None
            if enable_speed:
                measurement = speed_estimator.update(
                    tid,
                    (cx, cy),
                    frame_timestamp_seconds,
                    speed_line1.tolist(),
                    speed_line2.tolist(),
                    frame_index=frame_idx
                )

            speed = last_speed.get(tid, 0.0)

            if measurement and "speed_kmph" in measurement:
                speed = float(measurement["speed_kmph"])
                last_speed[tid] = speed

            speed_violation = enable_speed and speed > speed_config["speed_limit_kmph"]

            # ---------------- PARKING ----------------
            stationary = enable_parking and is_stationary(vehicle_positions[tid])

            in_zone = enable_parking and (
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

                    logger.info(
                        "Violation detected | Type=%s | ID=%s | Speed=%s km/h",
                        vtype,
                        tid,
                        int(speed)
                    )

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

                    # Write a shared rolling pre-event buffer instead of per-track copies.
                    for f in pre_event_buffer:
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
                cleanup_track(
                    tid,
                    vehicle_positions,
                    vehicle_entry_time,
                    last_speed,
                    locked_state,
                    recording_active,
                    violation_triggered,
                    speed_estimator
                )

        # =========================================================
        # UI
        # =========================================================
        if enable_speed:
            draw_speed_zones(annotated, speed_line1, speed_line2)
        if enable_parking:
            draw_parking_zones(annotated, parking_zone, buffer_zone)

        draw_status_hud(
            annotated,
            total_violations,
            f"Speed Limit {speed_config['speed_limit_kmph']} km/h" if enable_speed else "Parking Monitor",
            speed_violation_count,
            parking_violation_count
        )

        cv2.imshow("Smart Zebra", annotated)

        frame_idx += 1
        frame_idx = sync_video_to_realtime(
            cap,
            fps,
            frame_idx,
            playback_start_time,
            realtime_sync_enabled
        )

        wait_delay_ms = 1
        if realtime_sync_enabled:
            next_frame_time = playback_start_time + (frame_idx / fps)
            remaining_seconds = next_frame_time - time.perf_counter()
            if remaining_seconds > 0:
                wait_delay_ms = max(1, int(remaining_seconds * 1000))

        if cv2.waitKey(wait_delay_ms) == 27:
            break

    cap.release()

    for w in recording_active.values():
        w.release()

    cv2.destroyAllWindows()


# =========================================================
if __name__ == "__main__":
    main()
