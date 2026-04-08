import cv2
import json
import os
import csv
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

from tracker import update_tracker
from utils import get_bottom_center
from speed_estimator import SpeedEstimator

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "speed_config.json"
VEHICLE_CLASSES = {1,2,3,5,7}


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

    if "line1_y" in config and "line2_y" in config:
        line1_y = int(config["line1_y"])
        line2_y = int(config["line2_y"])
        return ((0, line1_y), (frame_width - 1, line1_y)), ((0, line2_y), (frame_width - 1, line2_y))

    raise KeyError("speed_config.json must define either line1/line2 or line1_y/line2_y")


def resolve_boxes(config):
    if "box1" not in config or "box2" not in config:
        raise KeyError("speed_config.json must define both box1 and box2")

    box1 = tuple(map(tuple, config["box1"]))
    box2 = tuple(map(tuple, config["box2"]))
    return box1, box2


def scale_line(line, scale):
    return ((int(line[0][0]*scale), int(line[0][1]*scale)),
            (int(line[1][0]*scale), int(line[1][1]*scale)))


def draw_zone_overlay(frame, zone1, zone2, zone_type):
    if zone_type == "box":
        cv2.rectangle(frame, zone1[0], zone1[1], (0, 0, 255), 3)
        cv2.rectangle(frame, zone2[0], zone2[1], (255, 0, 0), 3)
    else:
        cv2.line(frame, zone1[0], zone1[1], (0,0,255), 3)
        cv2.line(frame, zone2[0], zone2[1], (255,0,0), 3)


def draw_label(frame, text, origin, color):
    x, y = origin
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    top = max(0, y - text_height - baseline - 8)
    cv2.rectangle(
        frame,
        (x, top),
        (x + text_width + 10, y),
        color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x + 5, y - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

def main():
    config = load_config()

    model = YOLO(config["model_path"], task=config.get("model_task", "detect"))
    cap = cv2.VideoCapture(config["video_path"])

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {config['video_path']}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    estimator = SpeedEstimator(fps, config["distance_meters"])
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
    if "box1" in config and "box2" in config:
        zone1, zone2 = resolve_boxes(config)
        zone_type = "box"
    else:
        zone1, zone2 = resolve_lines(config, frame_width)
        zone_type = "line"

    log_path = config["violations_log_path"]
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

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

        draw_zone_overlay(frame, zone1, zone2, zone_type)

        for tid,x1,y1,x2,y2,cls in tracks:
            pt = get_bottom_center(x1,y1,x2,y2)
            measurement, _events = estimator.update(
                tid, pt, frame_id, zone1, zone2, zone_type
            )

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            if measurement:
                speed = measurement["speed_kmph"]
                color = (0,0,255) if speed > config["speed_limit_kmph"] else (0,255,0)
                draw_label(frame, f"{speed:.1f} km/h", (x1, max(30, y1 - 12)), color)

                if speed > config["speed_limit_kmph"] and tid not in logged:
                    output_dir = PROJECT_ROOT / "outputs" / "violations"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    img_path = str(output_dir / f"{tid}.jpg")
                    cv2.imwrite(img_path, frame)

                    with open(log_path,"a",newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([datetime.now(), tid, speed, img_path])

                    logged.add(tid)

        cv2.imshow("Speed Detection", frame)
        if cv2.waitKey(frame_delay_ms) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
