import os
import cv2
import csv
import gc
from datetime import datetime
import json
from ultralytics import YOLO
from tracker import update_tracker
from utils import get_bottom_center
from speed_estimator import SpeedEstimator

# Base directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configuration file path
config_path = os.path.join(base_dir, "config", "speed_config.json")

VEHICLE_CLASSES = {1, 2, 3, 5, 7}
PERSON_CLASS = 0
MAX_PROCESS_WIDTH = 1280


def log(message):
    print(message, flush=True)


def load_config():
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            log(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            log(f"Error loading config file: {e}")
            log("Using default configuration")

    return {
        "video_path": "videos/crossing1.mp4",
        "model_path": "models/best.pt",
        "output_video_path": "outputs/result.mp4",
        "violations_log_path": "outputs/violations/speeding_violations.csv",
        "speed_limit_kmph": 50,
        "line1_y": 322,
        "line2_y": 350,
        "distance_meters": 12,
    }


def existing_paths(*candidates):
    seen = set()
    for path in candidates:
        if not path:
            continue
        normalized = os.path.normpath(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        if os.path.exists(normalized):
            return normalized
    return None


def create_video_writer(output_path, fps, frame_width, frame_height):
    output_dir = os.path.dirname(output_path)
    output_name, _ = os.path.splitext(os.path.basename(output_path))

    candidates = [
        (os.path.join(output_dir, f"{output_name}.avi"), "XVID"),
        (os.path.join(output_dir, f"{output_name}_mjpg.avi"), "MJPG"),
        (output_path, "mp4v"),
    ]

    for candidate_path, codec in candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(candidate_path, fourcc, fps, (frame_width, frame_height))
        if writer.isOpened():
            log(f"Saving output video to: {candidate_path} using codec {codec}")
            return writer, candidate_path
        writer.release()

    raise RuntimeError(f"Could not create an output video writer for {output_path}.")


def resize_for_processing(frame, max_width):
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame, 1.0

    scale = max_width / float(width)
    new_size = (int(width * scale), int(height * scale))
    resized = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
    return resized, scale


def resolve_model_path(config):
    configured_model = config.get("model_path", "")
    candidates = [
        configured_model,
        os.path.join(base_dir, configured_model),
        os.path.join(base_dir, "models", "best.pt"),
        os.path.join(base_dir, "yolov8n.pt"),
    ]

    model_path = existing_paths(*candidates)
    if model_path:
        log(f"Using model: {model_path}")
        return model_path

    raise FileNotFoundError(
        "No YOLO model file found. Checked config path, models/best.pt, and yolov8n.pt."
    )


def resolve_video_path(config):
    configured_video = config.get("video_path", "")
    candidates = [
        configured_video,
        os.path.join(base_dir, configured_video),
        os.path.join(base_dir, "videos", "crossing1.mp4"),
        os.path.join(base_dir, "videos", "vid1.mp4"),
        os.path.join(base_dir, "videos", "videoplayback.mp4"),
    ]

    video_path = existing_paths(*candidates)
    if video_path:
        log(f"Using video: {video_path}")
        return video_path

    raise FileNotFoundError(
        "No input video found. Checked config path and common files in videos/."
    )


def main():
    config = load_config()

    try:
        model_path = resolve_model_path(config)
        video_path = resolve_video_path(config)
    except FileNotFoundError as exc:
        log(f"Error: {exc}")
        raise SystemExit(1)

    try:
        model = YOLO(model_path)
        log(f"Successfully loaded YOLO model from {model_path}")
    except Exception as exc:
        log(f"Error loading YOLO model: {exc}")
        raise SystemExit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"Error: Could not open video file {video_path}")
        raise SystemExit(1)

    log(f"Successfully opened video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    log(f"Video FPS: {fps}")

    configured_output_path = os.path.join(base_dir, config["output_video_path"])
    log_path = os.path.join(base_dir, config["violations_log_path"])
    output_dir = os.path.dirname(configured_output_path)
    violations_dir = os.path.dirname(log_path)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(violations_dir, exist_ok=True)

    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    process_scale = 1.0
    frame_width = source_width
    frame_height = source_height

    if source_width > MAX_PROCESS_WIDTH:
        process_scale = MAX_PROCESS_WIDTH / float(source_width)
        frame_width = int(source_width * process_scale)
        frame_height = int(source_height * process_scale)
        log(
            f"Processing resized frames at {frame_width}x{frame_height} "
            f"from source {source_width}x{source_height}"
        )

    try:
        out, output_path = create_video_writer(configured_output_path, fps, frame_width, frame_height)
    except RuntimeError as exc:
        log(f"Error: {exc}")
        cap.release()
        raise SystemExit(1)

    line1 = int(config["line1_y"] * process_scale)
    line2 = int(config["line2_y"] * process_scale)
    speed_limit = config["speed_limit_kmph"]
    estimator = SpeedEstimator(fps=fps, distance_meters=config["distance_meters"])
    frame_count = 0
    show_window = True

    if not os.path.exists(log_path):
        with open(log_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Vehicle_ID", "Speed_KMPH", "Snapshot_File"])

    logged_violations = set()

    while True:
        try:
            ret, frame = cap.read()
        except cv2.error as exc:
            log(f"OpenCV failed while reading a frame: {exc}")
            break

        if not ret:
            break

        frame, _ = resize_for_processing(frame, MAX_PROCESS_WIDTH)

        results = model(frame, imgsz=640, conf=0.25, max_det=50, verbose=False)[0]
        detections = []

        for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls)

            if class_id not in VEHICLE_CLASSES and class_id != PERSON_CLASS:
                continue

            detections.append(([x1, y1, x2 - x1, y2 - y1], conf.item(), class_id))

        tracks = update_tracker(detections, frame)

        cv2.line(frame, (0, line1), (frame_width, line1), (0, 0, 255), 2)
        cv2.line(frame, (0, line2), (frame_width, line2), (255, 0, 0), 2)
        cv2.putText(frame, "Line 1", (20, line1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, "Line 2", (20, line2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        for track_id, x1, y1, x2, y2, class_id in tracks:
            _, cy = get_bottom_center(x1, y1, x2, y2)
            speed_kmph = estimator.update(track_id, cy, frame_count, line1, line2)

            shrink = 5
            x1 += shrink
            y1 += shrink
            x2 -= shrink
            y2 -= shrink

            if class_id == PERSON_CLASS:
                color = (255, 0, 0)
                label = "Pedestrian"
            else:
                color = (0, 255, 0)
                label = "Vehicle"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label} ID {track_id}",
                (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

            if class_id in VEHICLE_CLASSES and speed_kmph is not None:
                is_speeding = speed_kmph > speed_limit
                display_color = (0, 0, 255) if is_speeding else color

                if is_speeding:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                    cv2.putText(
                        frame,
                        "SPEEDING!!",
                        (x1, y1 - 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        3,
                    )

                    snapshot_filename = f"vehicle_{track_id}_speed_{int(speed_kmph)}.jpg"
                    snapshot_path = os.path.join(base_dir, "outputs", "violations", snapshot_filename)

                    if track_id not in logged_violations:
                        cv2.imwrite(snapshot_path, frame)
                        with open(log_path, mode="a", newline="") as f:
                            writer = csv.writer(f)
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            writer.writerow([timestamp, track_id, f"{speed_kmph:.1f}", snapshot_filename])

                        logged_violations.add(track_id)
                        log(f"Violation logged: Vehicle {track_id} at {speed_kmph:.1f} km/h")

                cv2.putText(
                    frame,
                    f"Speed: {speed_kmph:.1f} km/h",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    display_color,
                    2,
                )

        out.write(frame)
        frame_count += 1

        if show_window:
            try:
                cv2.imshow("Speed Detection", frame)
                if cv2.waitKey(1) == 27:
                    break
            except cv2.error:
                show_window = False
                log("OpenCV display window is unavailable. Continuing to save the output video.")

        del results
        del detections
        del tracks
        del frame

        if frame_count % 60 == 0:
            gc.collect()

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    log(f"Finished. Output saved to: {output_path}")
    log(f"Violations log saved to: {log_path}")


if __name__ == "__main__":
    main()
