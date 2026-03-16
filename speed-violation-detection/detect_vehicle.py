import os
import cv2
import csv
from datetime import datetime
from ultralytics import YOLO
from tracker import update_tracker
from utils import get_bottom_center
from speed_estimator import SpeedEstimator

# Base directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load YOLO model
model_path = os.path.join(base_dir, "models", "best.pt")
model = YOLO(model_path)

# Load video
video_path = os.path.join(base_dir, "videos", "crossing1.mp4")
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
print("Video FPS:", fps)

# Output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_path = os.path.join(base_dir, "outputs", "result.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Speed detection lines
LINE1 = 322
LINE2 = 350

estimator = SpeedEstimator(fps=fps, distance_meters=12)
frame_count = 0

# Initialize Violation Log
log_path = os.path.join(base_dir, "outputs", "violations", "speeding_violations.csv")
if not os.path.exists(log_path):
    with open(log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Vehicle_ID", "Speed_KMPH", "Snapshot_File"])

logged_violations = set() # Track which IDs we already logged to avoid duplicates

while True:

    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=640)[0]

    detections = []

    for box, conf, cls in zip(results.boxes.xyxy,
                              results.boxes.conf,
                              results.boxes.cls):

        x1, y1, x2, y2 = map(int, box)
        class_id = int(cls)

        detections.append(([x1, y1, x2-x1, y2-y1], conf.item(), class_id))

    tracks = update_tracker(detections, frame)

    # Draw speed detection lines
    cv2.line(frame,(0,LINE1),(frame_width,LINE1),(0,0,255),2)
    cv2.line(frame,(0,LINE2),(frame_width,LINE2),(255,0,0),2)

    cv2.putText(frame,"Line 1",(20,LINE1-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

    cv2.putText(frame,"Line 2",(20,LINE2-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)

    for track_id, x1, y1, x2, y2, class_id in tracks:

        cx, cy = get_bottom_center(x1, y1, x2, y2)

        # Compute speed using line-crossing method
        speed_kmph = estimator.update(track_id, cy, frame_count, LINE1, LINE2)

        shrink = 5
        x1 += shrink
        y1 += shrink
        x2 -= shrink
        y2 -= shrink

        if class_id == 0:
            color = (255,0,0)
            label = "Pedestrian"
        else:
            color = (0,255,0)
            label = "Vehicle"

        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

        cv2.putText(
            frame,
            f"{label} ID {track_id}",
            (x1,y1-30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        # Show speed only for vehicles
        if class_id == 1 and speed_kmph is not None:
            SPEED_LIMIT = 50 # threshold for violation
            
            is_speeding = speed_kmph > SPEED_LIMIT
            display_color = (0, 0, 255) if is_speeding else color # Red if speeding
            
            # If speeding, draw a bolder rectangle and a "VIOLATION" label
            if is_speeding:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.putText(frame, "SPEEDING!!", (x1, y1 - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                
                # Save snapshot for the first time a vehicle is caught speeding
                snapshot_filename = f"vehicle_{track_id}_speed_{int(speed_kmph)}.jpg"
                snapshot_path = os.path.join(base_dir, "outputs", "violations", snapshot_filename)
                
                if track_id not in logged_violations:
                    cv2.imwrite(snapshot_path, frame)
                    
                    # Log to CSV
                    with open(log_path, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        writer.writerow([timestamp, track_id, f"{speed_kmph:.1f}", snapshot_filename])
                    
                    logged_violations.add(track_id)
                    print(f"Violation Logged: Vehicle {track_id} at {speed_kmph:.1f} km/h")

            cv2.putText(
                frame,
                f"Speed: {speed_kmph:.1f} km/h",
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                display_color,
                2
            )

    out.write(frame)
    frame_count += 1

    display = frame
    cv2.imshow("Speed Detection",display)

    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()