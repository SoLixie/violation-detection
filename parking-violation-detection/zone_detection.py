import cv2
import os
import numpy as np
import json

MIN_ZONE_POINTS = 3

# -------------------------------
# Paths
# -------------------------------
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, "config", "parking_config.json")

# -------------------------------
# Load / Save Config
# -------------------------------
def load_config():
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {
        "video_path": "videoplayback.mp4",
        "zebra_zone": [],
        "buffer_zone": [],
        "parking_threshold": 10
    }

def save_config(config):
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print("Zones saved successfully!")

# -------------------------------
# Video + Frame
# -------------------------------
def get_video_path(config):
    video_path = os.path.join(base_dir, config["video_path"])
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        exit()
    return video_path

def load_frame(video_path):
    cap = cv2.VideoCapture(video_path)

    for i in range(0, 150, 50):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret and np.mean(frame) > 10:
            cap.release()
            print(f"Using frame {i}")
            return frame

    cap.release()
    print("Could not extract frame")
    exit()

# -------------------------------
# Drawing
# -------------------------------
def draw_polygon(img, points, color, label=None):
    if len(points) >= MIN_ZONE_POINTS:
        overlay = img.copy()
        pts = np.array(points, np.int32)
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)

    for i, (x, y) in enumerate(points):
        cv2.circle(img, (x, y), 5, color, -1)
        cv2.putText(img, str(i+1), (x+5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if len(points) >= 2:
        pts = np.array(points, np.int32)
        cv2.polylines(img, [pts], len(points) >= MIN_ZONE_POINTS, color, 2)

    if label and len(points) >= MIN_ZONE_POINTS:
        cv2.putText(img, label, (points[0][0], points[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# -------------------------------
# Main
# -------------------------------
def main():
    config = load_config()
    video_path = get_video_path(config)
    frame = load_frame(video_path)

    zebra_points = []
    buffer_points = []

    mode = "ZEBRA"  # switch later

    existing_zebra = [tuple(p) for p in config.get("zebra_zone", [])]
    existing_buffer = [tuple(p) for p in config.get("buffer_zone", [])]

    def mouse(event, x, y, flags, param):
        nonlocal zebra_points, buffer_points, mode

        if event == cv2.EVENT_LBUTTONDOWN:
            if mode == "ZEBRA":
                zebra_points.append((x, y))
            else:
                buffer_points.append((x, y))

        elif event == cv2.EVENT_RBUTTONDOWN:
            if mode == "ZEBRA" and zebra_points:
                zebra_points.pop()
            elif mode == "BUFFER" and buffer_points:
                buffer_points.pop()

    cv2.namedWindow("Select Zones", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Select Zones", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback("Select Zones", mouse)

    print("\nInstructions:")
    print("Draw ZEBRA zone first")
    print("Press N → switch to BUFFER zone")
    print("Press S → save both zones")
    print("R → reset current zone | Q → quit\n")

    while True:
        display = frame.copy()

        # Existing zones
        if existing_zebra:
            draw_polygon(display, existing_zebra, (0, 255, 255), "CURRENT ZEBRA")
        if existing_buffer:
            draw_polygon(display, existing_buffer, (255, 255, 0), "CURRENT BUFFER")

        # New zones
        draw_polygon(display, zebra_points, (0, 255, 0), "ZEBRA (DRAWING)")
        draw_polygon(display, buffer_points, (255, 0, 0), "BUFFER (DRAWING)")

        cv2.putText(display, f"MODE: {mode}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Select Zones", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('n'):
            mode = "BUFFER"
            print("Switched to BUFFER zone")

        elif key == ord('r'):
            if mode == "ZEBRA":
                zebra_points = []
            else:
                buffer_points = []
            print("Reset current zone")

        elif key == ord('s'):
            if len(zebra_points) >= MIN_ZONE_POINTS and len(buffer_points) >= MIN_ZONE_POINTS:
                config["zebra_zone"] = [list(p) for p in zebra_points]
                config["buffer_zone"] = [list(p) for p in buffer_points]
                save_config(config)
                break
            else:
                print("Both zones need at least 3 points")

        elif key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()

# -------------------------------
if __name__ == "__main__":
    main()