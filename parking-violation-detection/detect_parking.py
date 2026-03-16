import cv2
import os
import time
import numpy as np
from ultralytics import YOLO
from tracker import update_tracker
from utils import get_centroid  # As per utils.py we have get_centroid and get_bottom_center

# Base directory (1 level up from this script's directory)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load YOLO model (already in root directory)
model_path = os.path.join(base_dir, "models", "best.pt")
model = YOLO(model_path)

# Video Input
video_path = os.path.join(base_dir, "videos", "illegal-parking.mp4")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# No Parking Zone Polygon (Zebra Crossing identification)
ZONE_POLYGON = np.array([
    [292, 271], [101, 416], 
    [545, 588], [938, 444]
], np.int32)

# Parking threshold (in seconds)
PARKING_THRESHOLD = 10

# Tracking data
vehicle_entry_time = {} # {track_id: start_time}
violation_count = 0

def is_inside_zone(cx, cy):
    """Check if the point (cx, cy) is inside the defined polygon zone."""
    result = cv2.pointPolygonTest(ZONE_POLYGON, (float(cx), float(cy)), False)
    return result >= 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection with a lower confidence threshold (e.g., 0.15 or 0.1)
    results = model(frame, imgsz=640, conf=0.15, verbose=False)[0]

    # Define target classes
    VEHICLE_CLASSES = [1, 2, 3, 5, 7] 
    PERSON_CLASS = 0

    detections = []
    for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        class_id = int(cls)
        
        # Track both vehicles and pedestrians
        if class_id in VEHICLE_CLASSES or class_id == PERSON_CLASS:
            detections.append(([x1, y1, x2-x1, y2-y1], conf.item(), class_id))

    # Update tracker
    tracks = update_tracker(detections, frame)

    # Draw the NO PARKING ZONE (Polygon)
    cv2.polylines(frame, [ZONE_POLYGON], True, (255, 255, 0), 2)
    cv2.putText(frame, "NO PARKING ZONE", (ZONE_POLYGON[0][0], ZONE_POLYGON[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    current_time = time.time()

    for track_id, x1, y1, x2, y2, class_id in tracks:
        # Compute centroid using utils.py helper
        cx, cy = get_centroid(x1, y1, x2, y2)
        
        # Default styling
        is_vehicle = class_id in VEHICLE_CLASSES
        color = (0, 255, 0) if is_vehicle else (255, 0, 255) # Green for cars, Purple for people
        label = "Vehicle" if is_vehicle else "Pedestrian"
        duration = 0

        # PARKING VIOLATION LOGIC (ONLY FOR VEHICLES)
        if is_vehicle:
            # Check if vehicle is inside the zone
            inside = is_inside_zone(cx, cy)
            
            if inside:
                if track_id not in vehicle_entry_time:
                    vehicle_entry_time[track_id] = current_time
                
                duration = current_time - vehicle_entry_time[track_id]
                
                if duration > PARKING_THRESHOLD:
                    color = (0, 0, 255) # Warning Red
                    label = "ILLEGAL PARKING"
                    cv2.putText(frame, label, (x1, y1 - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                cv2.putText(frame, f"Time: {int(duration)}s", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                if track_id in vehicle_entry_time:
                    del vehicle_entry_time[track_id]
        
        # Visualization (Shown for both Pedestrians and Vehicles)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (cx, cy), 5, color, -1)
        cv2.putText(frame, f"{label} {track_id}", (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the frame
    cv2.imshow("Illegal Parking Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27: # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
