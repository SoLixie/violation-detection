import cv2
import json
import os
import csv
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import sys

try:
    from .speed_estimator import SpeedEstimator
except ImportError:
    from speed_estimator import SpeedEstimator

# Import from canonical locations (NO DUPLICATION)
sys.path.insert(0, str(Path(__file__).parent.parent))
from tracker import update_tracker
from common.geometry import get_bottom_center
from storage.mongo_handler import get_mongo_handler
from visual_utils import draw_styled_line, draw_vehicle_box, draw_label, draw_status_hud

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "speed_config.json"
VEHICLE_CLASSES = {1,2,3,5,7}
CLASS_LABELS = {
    1: "Bicycle",
    2: "Car",
    3: "Motorbike",
    5: "Bus",
    7: "Truck",
}
LINE_STYLES = (
    {"name": "Entry", "color": (64, 92, 255)},
    {"name": "Exit", "color": (255, 191, 64)},
)
ROAD_MARKING_PRESETS = {
    "asian_national_highways": {"road_marking_meters": 6.0, "gap_meters": 3.0, "line_width_mm": 100},
    "thromde": {"road_marking_meters": 3.0, "gap_meters": 1.5, "line_width_mm": 100},
    "dzongkhag_other_roads": {"road_marking_meters": 4.0, "gap_meters": 2.0, "line_width_mm": 100},
}
UI_FONT = cv2.FONT_HERSHEY_DUPLEX


def require_env_path(var_name):
    value = os.getenv(var_name, "").strip()
    if not value:
        raise RuntimeError(
            f"Set the {var_name} environment variable to the external SSD folder path."
        )

    path = Path(value)
    if not path.exists():
        raise RuntimeError(
            f"{var_name} points to '{path}', but that path does not exist. "
            "Connect the external SSD and try again."
        )

    if not path.is_dir():
        raise RuntimeError(f"{var_name} must point to a directory, not a file: {path}")

    return path


def require_env_value(var_name, help_text):
    value = os.getenv(var_name, "").strip()
    if not value:
        raise RuntimeError(f"Set the {var_name} environment variable. {help_text}")
    return value


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = json.load(f)

    for key in ("video_path", "model_path", "output_video_path", "violations_log_path"):
        if key not in config:
            continue

        value = config[key]
        if key == "model_path":
            model_path = Path(value)
            if model_path.is_absolute() or model_path.parent != Path('.'):
                config[key] = str((PROJECT_ROOT / value).resolve())
            else:
                config[key] = value
            continue

        path_value = Path(value)
        if key == "video_path" and not path_value.is_absolute():
            if path_value.parts and path_value.parts[0] == "videos":
                legacy_candidate = PROJECT_ROOT / "vids" / Path(*path_value.parts[1:])
                if legacy_candidate.exists():
                    config[key] = str(legacy_candidate.resolve())
                    continue

        config[key] = str((PROJECT_ROOT / value).resolve())

    return config


def resolve_lines(config, frame_width):
    if "line1" in config and "line2" in config:
        line1 = tuple(map(tuple, config["line1"]))
        line2 = tuple(map(tuple, config["line2"]))
        return line1, line2
    raise KeyError("speed_config.json must define both line1 and line2")


def resolve_distance_settings(config):
    road_type = config.get("road_type")
    measurement_basis = config.get("measurement_basis", "distance_meters")

    if road_type:
        if road_type not in ROAD_MARKING_PRESETS:
            valid = ", ".join(ROAD_MARKING_PRESETS)
            raise KeyError(f"Unsupported road_type '{road_type}'. Use one of: {valid}")
        preset = ROAD_MARKING_PRESETS[road_type]
        config.setdefault("road_marking_meters", preset["road_marking_meters"])
        config.setdefault("gap_meters", preset["gap_meters"])
        config.setdefault("line_width_mm", preset["line_width_mm"])

    if measurement_basis == "distance_meters":
        if "distance_meters" not in config:
            raise KeyError("speed_config.json must define distance_meters when measurement_basis is 'distance_meters'")
        return float(config["distance_meters"]), f"{float(config['distance_meters']):g} m measured distance"

    if measurement_basis == "road_marking":
        if "road_marking_meters" not in config:
            raise KeyError("speed_config.json must define road_marking_meters when measurement_basis is 'road_marking'")
        distance = float(config["road_marking_meters"])
        return distance, f"{distance:g} m road marking"

    if measurement_basis == "gap":
        if "gap_meters" not in config:
            raise KeyError("speed_config.json must define gap_meters when measurement_basis is 'gap'")
        distance = float(config["gap_meters"])
        return distance, f"{distance:g} m marking gap"

    raise KeyError("measurement_basis must be one of: distance_meters, road_marking, gap")


def scale_line(line, scale):
    return ((int(line[0][0]*scale), int(line[0][1]*scale)),
            (int(line[1][0]*scale), int(line[1][1]*scale)))


def main():
    config = load_config()

    # Use unified mongo handler (CANONICAL MONGO HANDLER)
    try:
        mongo_handler = get_mongo_handler()
    except RuntimeError as e:
        print(f"Warning: MongoDB not available: {e}")
        mongo_handler = None

    ssd_root = require_env_path("VIOLATIONS_SSD_PATH")
    output_root = ssd_root / "speed_violations"
    image_output_dir = output_root / "images"
    image_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "speeding_violations.csv"

    model = YOLO(config["model_path"], task=config.get("model_task", "detect"))
    cap = cv2.VideoCapture(config["video_path"])

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {config['video_path']}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    distance_meters, _distance_label = resolve_distance_settings(config)
    estimator = SpeedEstimator(fps, distance_meters)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
    zone1, zone2 = resolve_lines(config, frame_width)

    cv2.namedWindow("Speed Detection", cv2.WINDOW_NORMAL)

    logged = set()
    frame_delay_ms = max(1, int(1000 / fps))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)

        results = model(frame, conf=0.25, imgsz=416, verbose=False)[0]
        detections = []

        for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            x1,y1,x2,y2 = map(int, box)
            if int(cls) not in VEHICLE_CLASSES:
                continue
            detections.append(([x1,y1,x2-x1,y2-y1], conf.item(), int(cls)))

        tracks = update_tracker(detections, frame)

        draw_speed_zones(frame, zone1, zone2)
        draw_status_hud(frame, len(logged), f"Limit {config['speed_limit_kmph']} km/h")

        for tid,x1,y1,x2,y2,cls in tracks:
            pt = get_bottom_center(x1,y1,x2,y2)
            measurement, _events = estimator.update(
                tid, pt, frame_id, zone1, zone2
            )

            label_x = int((x1 + x2) / 2)
            detail_y = y1 + 18 if y1 < 28 else y1 - 6
            if measurement:
                speed = measurement["speed_kmph"]
                is_violation = speed > config["speed_limit_kmph"]
                accent = (52, 211, 153) if not is_violation else (64, 92, 255)
                draw_vehicle_box(frame, (x1, y1, x2, y2), accent, emphasis=is_violation)
                speed_text = f"{speed:.1f} km/h"
                # Speed is only available after the tracker has crossed both lines and the
                # estimator has created a completed measurement.
                draw_label(frame, speed_text, (label_x, detail_y), accent, font_scale=0.4, align="center")

                if is_violation and tid not in logged:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_path = image_output_dir / f"speed_{tid}_{timestamp}.jpg"
                    cv2.imwrite(str(img_path), frame)

                    # Use unified mongo handler (CANONICAL HANDLER)
                    if mongo_handler:
                        try:
                            with open(img_path, "rb") as image_file:
                                image_bytes = image_file.read()
                            mongo_handler.save_violation(
                                track_id=tid,
                                vtype="speed",
                                speed=speed,
                                image_bytes=image_bytes,
                                metadata={
                                    "image_filename": img_path.name,
                                    "image_path": str(img_path),
                                    "camera_id": "speed_detection"
                                }
                            )
                        except Exception as exc:
                            print(f"Warning: Failed to save violation to MongoDB: {exc}")

                    with open(log_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([datetime.now(), tid, speed, str(img_path)])

                    logged.add(tid)
            else:
                accent = (125, 211, 252)
                draw_vehicle_box(frame, (x1, y1, x2, y2), accent)

        cv2.imshow("Speed Detection", frame)
        if cv2.waitKey(frame_delay_ms) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

