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


def draw_styled_line(frame, line, label, color):
    overlay = frame.copy()
    cv2.line(overlay, line[0], line[1], color, 7, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)

    cv2.line(frame, line[0], line[1], (255, 255, 255), 3, cv2.LINE_AA)
    cv2.line(frame, line[0], line[1], color, 2, cv2.LINE_AA)

    for point in line:
        cv2.circle(frame, point, 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, point, 3, color, -1, cv2.LINE_AA)

    side_point = line[0] if line[0][0] <= line[1][0] else line[1]
    anchor_x = side_point[0] + 12
    anchor_y = max(24, side_point[1] - 10)
    (text_width, text_height), baseline = cv2.getTextSize(
        label, UI_FONT, 0.45, 1
    )
    left = max(8, anchor_x)
    top = max(8, anchor_y - text_height - baseline - 6)
    right = left + text_width + 12
    bottom = top + text_height + baseline + 8

    panel = frame.copy()
    cv2.rectangle(panel, (left, top), (right, bottom), (20, 24, 32), -1)
    cv2.rectangle(panel, (left, top), (right, bottom), color, 1)
    cv2.addWeighted(panel, 0.52, frame, 0.48, 0, frame)

    cv2.putText(
        frame,
        label,
        (left + 6, bottom - baseline - 4),
        UI_FONT,
        0.45,
        (245, 247, 250),
        1,
        cv2.LINE_AA,
    )


def draw_zone_overlay(frame, line1, line2):
    draw_styled_line(frame, line1, LINE_STYLES[0]["name"], LINE_STYLES[0]["color"])
    draw_styled_line(frame, line2, LINE_STYLES[1]["name"], LINE_STYLES[1]["color"])


def draw_label(frame, text, origin, color, text_color=(255, 255, 255), font_scale=0.48, align="left"):
    frame_h, frame_w = frame.shape[:2]
    x, y = origin
    (text_width, text_height), baseline = cv2.getTextSize(
        text, UI_FONT, font_scale, 1
    )
    box_width = text_width + 12
    box_height = text_height + baseline + 8
    if align == "center":
        x -= box_width // 2
    x = max(6, min(x, frame_w - box_width - 6))
    y = max(box_height + 6, min(y, frame_h - 6))
    top = y - box_height

    panel = frame.copy()
    cv2.rectangle(panel, (x, top), (x + box_width, y), (16, 20, 28), -1)
    cv2.rectangle(panel, (x, top), (x + box_width, y), color, 1)
    cv2.addWeighted(panel, 0.58, frame, 0.42, 0, frame)

    cv2.putText(
        frame,
        text,
        (x + 6, y - 5),
        UI_FONT,
        font_scale,
        text_color,
        1,
        cv2.LINE_AA,
    )


def draw_vehicle_box(frame, box, color, emphasis=False):
    x1, y1, x2, y2 = box
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.03 if not emphasis else 0.06, frame, 0.97 if not emphasis else 0.94, 0, frame)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    corner = 10
    segments = (
        ((x1, y1), (x1 + corner, y1)),
        ((x1, y1), (x1, y1 + corner)),
        ((x2, y1), (x2 - corner, y1)),
        ((x2, y1), (x2, y1 + corner)),
        ((x1, y2), (x1 + corner, y2)),
        ((x1, y2), (x1, y2 - corner)),
        ((x2, y2), (x2 - corner, y2)),
        ((x2, y2), (x2, y2 - corner)),
    )
    for start, end in segments:
        cv2.line(frame, start, end, color, 1, cv2.LINE_AA)


def draw_status_hud(frame, speed_limit, violation_count):
    frame_w = frame.shape[1]
    panel = frame.copy()
    left = max(8, frame_w - 210)
    top = 12
    right = frame_w - 12
    bottom = 62
    cv2.rectangle(panel, (left, top), (right, bottom), (18, 22, 30), -1)
    cv2.addWeighted(panel, 0.56, frame, 0.44, 0, frame)

    cv2.putText(
        frame,
        f"Violations {violation_count}",
        (left + 12, top + 22),
        UI_FONT,
        0.55,
        (245, 247, 250),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Limit {speed_limit} km/h",
        (left + 12, top + 42),
        UI_FONT,
        0.43,
        (203, 213, 225),
        1,
        cv2.LINE_AA,
    )

def main():
    config = load_config()

    model = YOLO(config["model_path"], task=config.get("model_task", "detect"))
    cap = cv2.VideoCapture(config["video_path"])

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {config['video_path']}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    distance_meters, _distance_label = resolve_distance_settings(config)
    estimator = SpeedEstimator(fps, distance_meters)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
    zone1, zone2 = resolve_lines(config, frame_width)

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

        draw_zone_overlay(frame, zone1, zone2)
        draw_status_hud(frame, config["speed_limit_kmph"], len(logged))

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
                    output_dir = PROJECT_ROOT / "outputs" / "violations"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    img_path = str(output_dir / f"{tid}.jpg")
                    cv2.imwrite(img_path, frame)

                    with open(log_path,"a",newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([datetime.now(), tid, speed, img_path])

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
