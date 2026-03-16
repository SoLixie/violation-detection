import cv2
import os
import numpy as np

# Base directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
video_path = os.path.join(base_dir, "videos", "illegal-parking.mp4")

if not os.path.exists(video_path):
    print(f"Error: {video_path} not found.")
    exit()

# Load the video and get a clear frame
cap = cv2.VideoCapture(video_path)
frame = None
found_frame = False
for i in range(0, 150, 50): # Check frames 0, 50, 100
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, temp_frame = cap.read()
    if ret and np.mean(temp_frame) > 10: # Skip black frames
        frame = temp_frame
        found_frame = True
        print(f"Using frame {i} for identification.")
        break

if not found_frame:
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read video frame.")
        cap.release()
        exit()
cap.release()

image_display = frame.copy()
points = []

def mouse_callback(event, x, y, flags, param):
    global points, image_display, frame
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            # Draw the point
            cv2.circle(image_display, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(image_display, str(len(points)), (x + 10, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw line between points
            if len(points) > 1:
                cv2.line(image_display, points[-2], points[-1], (0, 255, 0), 2)
            
            # Close the polygon
            if len(points) == 4:
                cv2.line(image_display, points[3], points[0], (0, 255, 0), 2)
                
                # Calculate bounding box
                pts = np.array(points)
                bx, by, bw, bh = cv2.boundingRect(pts)
                
                print(f"\nNo Parking Zone Coordinates (Bounding Box):")
                print(f"ZONE_X1 = {bx}")
                print(f"ZONE_Y1 = {by}")
                print(f"ZONE_X2 = {bx + bw}")
                print(f"ZONE_Y2 = {by + bh}")
                print(f"\nCoordinates for Polygon Detection (more accurate):")
                print(f"POLYGON = {points}")
                print(f"\nUpdate these in detect_parking_violation.py.")
                print("Press 'q' to exit or 'r' to reset.")

        cv2.imshow("Select Parking Zone", image_display)

cv2.namedWindow("Select Parking Zone", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Select Parking Zone", mouse_callback)

print("Instructions:")
print("1. Click 4 points to define the NO PARKING ZONE.")
print("2. The bounding box will be automatically calculated.")
print("3. Press 'r' to reset.")
print("4. Press 'q' or 'ESC' to exit.")

while True:
    cv2.imshow("Select Parking Zone", image_display)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break
    elif key == ord('r'):
        points = []
        image_display = frame.copy()
        print("Resetting...")

cv2.destroyAllWindows()
