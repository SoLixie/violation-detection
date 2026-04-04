import os
import time
import json
import sys
import importlib.util
import argparse

def log(message):
    print(message, flush=True)

# -------------------------------
# Base directory
# -------------------------------
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, "config", "parking_config.json")
output_video_path = os.path.join(base_dir, "outputs", "parking_detection_output.mp4")
display_max_width = 1280

# -------------------------------
# Load Config
# -------------------------------
def load_config():
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            log(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            log(f"Error loading config: {e}")

    return {
        "video_path": "videos/illegal-parking.mp4",
        "model_path": "models/best.pt",
        "zebra_zone": [],
        "buffer_zone": [],
        "parking_threshold": 10
    }

config = load_config()

# -------------------------------
# Resolve Paths
# -------------------------------
def resolve_model_path():
    candidates = [
        os.path.join(base_dir, config.get("model_path", "")),
        os.path.join(base_dir, "models", "best.pt")
    ]
    for path in candidates:
        if os.path.exists(path):
            log(f"Using model: {path}")
            return path

    log("Model not found.")
    exit(1)

def resolve_video_path():
    candidates = [
        os.path.join(base_dir, config.get("video_path", "")),
        config.get("video_path", "")
    ]
    for path in candidates:
        if os.path.exists(path):
            log(f"Using video: {path}")
            return path

    log("Video not found.")
    exit(1)

# -------------------------------
# Helpers
# -------------------------------
def is_inside_polygon(cx, cy, polygon):
    import cv2
    return cv2.pointPolygonTest(polygon, (float(cx), float(cy)), False) >= 0

def is_stationary(positions, threshold=5):
    import numpy as np
    if len(positions) < 5:
        return False

    distances = [
        np.linalg.norm(np.array(positions[i]) - np.array(positions[i-1]))
        for i in range(1, len(positions))
    ]
    return np.mean(distances) < threshold

def resize_for_output(frame, max_width):
    import cv2

    height, width = frame.shape[:2]
    if width <= max_width:
        return frame

    scale = max_width / float(width)
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

# -------------------------------
# Main
# -------------------------------
def main():
    log("Importing OpenCV...")
    import cv2
    log("OpenCV imported.")

    log("Importing NumPy...")
    import numpy as np
    log("NumPy imported.")

    log("Importing Ultralytics YOLO...")
    from ultralytics import YOLO
    log("Ultralytics imported.")

    log("Importing tracker...")
    from tracker import update_tracker
    log("Tracker imported.")

    log("Importing utils...")
    from utils import get_bottom_center
    log("Utils imported.")

    log("Starting detect_parking.py")
    model_path = resolve_model_path()
    video_path = resolve_video_path()

    zebra_zone = config.get("zebra_zone", [])
    buffer_zone = config.get("buffer_zone", [])

    if len(zebra_zone) < 3 or len(buffer_zone) < 3:
        log("ERROR: Configure both zebra and buffer zones first.")
        exit()

    zebra_polygon = np.array(zebra_zone, np.int32)
    buffer_polygon = np.array(buffer_zone, np.int32)

    log(f"Parking threshold: {config['parking_threshold']} seconds")
    log("Loading YOLO model...")
    model = YOLO(model_path)
    log("YOLO model loaded.")
    log("Opening video...")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        log("ERROR: OpenCV could not open the video file.")
        exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    preview_width = min(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), display_max_width)
    preview_height = int(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * (preview_width / float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))))

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (preview_width, preview_height)
    )

    show_window = True
    log("Starting parking detection. Press ESC to close the window.")
    log(f"Saving annotated output to: {output_video_path}")
    log(f"Display/output width capped at: {display_max_width}px")

    vehicle_entry_time = {}
    vehicle_positions = {}
    violated_ids = set()
    violation_count = 0
    frame_count = 0

    VEHICLE_CLASSES = [1, 2, 3, 5, 7]

    while True:
        ret, frame = cap.read()
        if not ret:
            log("Reached end of video or failed to read a frame.")
            break

        frame_count += 1
        if frame_count == 1:
            log("First video frame read successfully.")
        elif frame_count % 30 == 0:
            log(f"Processed {frame_count} frames...")

        results = model(frame, imgsz=640, conf=0.15, verbose=False)[0]

        detections = []
        for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls)

            if class_id in VEHICLE_CLASSES:
                detections.append(([x1, y1, x2-x1, y2-y1], conf.item(), class_id))

        tracks = update_tracker(detections, frame)

        # -------------------------------
        # Draw Zones
        # -------------------------------
        cv2.polylines(frame, [zebra_polygon], True, (0, 0, 255), 2)
        cv2.putText(frame, "ZEBRA ZONE",
                    (zebra_polygon[0][0], zebra_polygon[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.polylines(frame, [buffer_polygon], True, (0, 165, 255), 2)
        cv2.putText(frame, "BUFFER ZONE",
                    (buffer_polygon[0][0], buffer_polygon[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        current_time = time.time()

        for track_id, x1, y1, x2, y2, class_id in tracks:

            cx, cy = get_bottom_center(x1, y1, x2, y2)

            if track_id not in vehicle_positions:
                vehicle_positions[track_id] = []

            vehicle_positions[track_id].append((cx, cy))
            if len(vehicle_positions[track_id]) > 10:
                vehicle_positions[track_id].pop(0)

            stationary = is_stationary(vehicle_positions[track_id])
            inside_zebra = is_inside_polygon(cx, cy, zebra_polygon)
            inside_buffer = is_inside_polygon(cx, cy, buffer_polygon)

            color = (0, 255, 0)
            label = f"Vehicle {track_id}"

            if stationary and (inside_zebra or inside_buffer):

                if track_id not in vehicle_entry_time:
                    vehicle_entry_time[track_id] = current_time

                duration = current_time - vehicle_entry_time[track_id]

                if duration > config["parking_threshold"]:

                    if inside_zebra:
                        color = (0, 0, 255)
                        label = "ILLEGAL PARKING (ZEBRA)"
                    else:
                        color = (0, 165, 255)
                        label = "ILLEGAL PARKING (NEAR)"

                    if track_id not in violated_ids:
                        violation_count += 1
                        violated_ids.add(track_id)

                    cv2.putText(frame, label, (x1, y1 - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                cv2.putText(frame, f"Time: {int(duration)}s",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            else:
                if track_id in vehicle_entry_time:
                    del vehicle_entry_time[track_id]

            # Draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 5, color, -1)
            cv2.putText(frame, label, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(frame, f"Violations: {violation_count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        output_frame = resize_for_output(frame, display_max_width)
        writer.write(output_frame)

        if show_window:
            try:
                cv2.imshow("Illegal Parking Detection", output_frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            except cv2.error:
                show_window = False
                log("OpenCV display window is unavailable in this environment. Continuing to save the output video instead.")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    log(f"Finished. Output saved to: {output_video_path}")

# -------------------------------
if __name__ == "__main__":
    main()
