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
        if key in config:
            config[key] = str((PROJECT_ROOT / config[key]).resolve())

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


def scale_line(line, scale):
    return ((int(line[0][0]*scale), int(line[0][1]*scale)),
            (int(line[1][0]*scale), int(line[1][1]*scale)))


def main():
    config = load_config()

    if not os.path.exists(config["model_path"]):
        raise FileNotFoundError(
            f"Configured model file was not found: {config['model_path']}. "
            "Update config/speed_config.json to point to an existing model export."
        )

    model = YOLO(config["model_path"], task=config.get("model_task", "detect"))
    cap = cv2.VideoCapture(config["video_path"])

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {config['video_path']}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    estimator = SpeedEstimator(fps, config["distance_meters"])
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
    line1, line2 = resolve_lines(config, frame_width)

    log_path = config["violations_log_path"]
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logged = set()
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.25)[0]
        detections = []

        for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            x1,y1,x2,y2 = map(int, box)
            if int(cls) not in VEHICLE_CLASSES:
                continue
            detections.append(([x1,y1,x2-x1,y2-y1], conf.item(), int(cls)))

        tracks = update_tracker(detections, frame)

        cv2.line(frame, line1[0], line1[1], (0,0,255),2)
        cv2.line(frame, line2[0], line2[1], (255,0,0),2)

        for tid,x1,y1,x2,y2,cls in tracks:
            pt = get_bottom_center(x1,y1,x2,y2)
            speed = estimator.update(tid, pt, frame_id, line1, line2)

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            if speed:
                color = (0,0,255) if speed > config["speed_limit_kmph"] else (0,255,0)

                cv2.putText(frame,f"{speed:.1f} km/h",(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

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
        frame_id += 1

        if cv2.waitKey(1)==27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
