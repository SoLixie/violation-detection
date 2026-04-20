import os
import time
import json
import sys
import importlib.util
import argparse
from pathlib import Path
import cv2

def log(message):
    print(message, flush=True)

# -------------------------------
# Base directory
# -------------------------------
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, "config", "parking_config.json")
output_video_path = os.path.join(base_dir, "outputs", "parking_detection_output.mp4")
display_max_width = 1280
UI_FONT = cv2.FONT_HERSHEY_DUPLEX
CLASS_LABELS = {
    1: "Bicycle",
    2: "Car",
    3: "Motorbike",
    5: "Bus",
    7: "Truck",
}
ZONE_STYLES = {
    "zebra": {"label": "Zebra Zone", "color": (64, 92, 255)},
    "buffer": {"label": "Buffer Zone", "color": (255, 191, 64)},
}

# ========================
# CANONICAL IMPORTS (NO DUPLICATION)
# ========================
# Import from canonical locations instead of duplicating code
try:
    from tracker import update_tracker
    from common.geometry import get_bottom_center, is_inside_polygon, is_stationary
    from visual_utils import draw_badge, draw_styled_polygon, draw_vehicle_box, draw_status_hud
except ImportError:
    # Fallback for standalone execution
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from tracker import update_tracker
    from common.geometry import get_bottom_center, is_inside_polygon, is_stationary
    from visual_utils import draw_badge, draw_styled_polygon, draw_vehicle_box, draw_status_hud

# ========================

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
        "video_path": "vids/illegal-parking.mp4",
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

    video_path = config.get("video_path", "")
    if video_path.startswith("videos/"):
        candidates.append(os.path.join(base_dir, "vids", video_path.split("/", 1)[1]))

    for path in candidates:
        if os.path.exists(path):
            log(f"Using video: {path}")
            return path

    log("Video not found.")
    exit(1)


# -------------------------------
# Helpers - SHARED FROM COMMON (NO DUPLICATION)
# -------------------------------
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

    cv2.namedWindow("Illegal Parking Detection", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Illegal Parking Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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
        draw_styled_polygon(frame, zebra_polygon, ZONE_STYLES["zebra"]["label"], ZONE_STYLES["zebra"]["color"])
        draw_styled_polygon(frame, buffer_polygon, ZONE_STYLES["buffer"]["label"], ZONE_STYLES["buffer"]["color"])
        draw_status_hud(frame, violation_count, f"Threshold {config['parking_threshold']}s")

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

            color = (125, 211, 252)
            status_text = "Moving"
            label = f"{CLASS_LABELS.get(class_id, 'Vehicle')}"

            if stationary and (inside_zebra or inside_buffer):

                if track_id not in vehicle_entry_time:
                    vehicle_entry_time[track_id] = current_time

                duration = current_time - vehicle_entry_time[track_id]

                if duration > config["parking_threshold"]:

                    if inside_zebra:
                        color = ZONE_STYLES["zebra"]["color"]
                        status_text = "Violation  Zebra"
                    else:
                        color = ZONE_STYLES["buffer"]["color"]
                        status_text = "Violation  Buffer"

                    if track_id not in violated_ids:
                        violation_count += 1
                        violated_ids.add(track_id)
                else:
                    color = (52, 211, 153)
                    status_text = "Observed"

            else:
                if track_id in vehicle_entry_time:
                    del vehicle_entry_time[track_id]

            draw_vehicle_box(frame, (x1, y1, x2, y2), color, emphasis=track_id in violated_ids)
            cv2.circle(frame, (cx, cy), 8, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 4, color, -1, cv2.LINE_AA)
            title_y = y1 + 18 if y1 < 28 else y1 - 6
            center_x = int((x1 + x2) / 2)

            if track_id in vehicle_entry_time:
                duration = current_time - vehicle_entry_time[track_id]
                draw_badge(frame, f"{status_text} {int(duration)}s", (center_x - 34, title_y), color, font_scale=0.4)
            else:
                draw_badge(frame, status_text, (center_x - 24, title_y), color, font_scale=0.4)

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

