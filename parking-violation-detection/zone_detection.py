import cv2
import os
import numpy as np
import json

MIN_ZONE_POINTS = 3
UI_FONT = cv2.FONT_HERSHEY_DUPLEX
ZONE_STYLES = {
    "ZEBRA": {"label": "Zebra Zone", "color": (64, 92, 255)},
    "BUFFER": {"label": "Buffer Zone", "color": (255, 191, 64)},
}
EXISTING_ZONE_STYLES = {
    "ZEBRA": {"label": "Current Zebra", "color": (64, 224, 208)},
    "BUFFER": {"label": "Current Buffer", "color": (168, 85, 247)},
}

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
        cv2.addWeighted(overlay, 0.1, img, 0.9, 0, img)

    for i, (x, y) in enumerate(points):
        cv2.circle(img, (x, y), 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), 3, color, -1, cv2.LINE_AA)
        cv2.putText(img, str(i+1), (x+10, y-8),
                    UI_FONT, 0.42, (245, 247, 250), 1, cv2.LINE_AA)

    if len(points) >= 2:
        pts = np.array(points, np.int32)
        cv2.polylines(img, [pts], len(points) >= MIN_ZONE_POINTS, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.polylines(img, [pts], len(points) >= MIN_ZONE_POINTS, color, 2, cv2.LINE_AA)

    if label and len(points) >= MIN_ZONE_POINTS:
        anchor = min(points, key=lambda p: p[0])
        draw_badge(img, label, (anchor[0] + 10, max(26, anchor[1] - 8)), color, font_scale=0.45)


def draw_badge(img, text, origin, color, font_scale=0.6):
    img_h, img_w = img.shape[:2]
    x, y = origin
    (text_width, text_height), baseline = cv2.getTextSize(
        text, UI_FONT, font_scale, 1
    )
    box_width = text_width + 12
    box_height = text_height + baseline + 8
    x = max(6, min(x, img_w - box_width - 6))
    y = max(box_height + 6, min(y, img_h - 6))
    top = y - box_height
    right = x + box_width

    panel = img.copy()
    cv2.rectangle(panel, (x, top), (right, y), (16, 20, 28), -1)
    cv2.rectangle(panel, (x, top), (right, y), color, 1)
    cv2.addWeighted(panel, 0.58, img, 0.42, 0, img)

    cv2.putText(
        img,
        text,
        (x + 6, y - 5),
        UI_FONT,
        font_scale,
        (245, 247, 250),
        1,
        cv2.LINE_AA,
    )


def draw_header(img, mode):
    img_w = img.shape[1]
    panel = img.copy()
    left = max(8, img_w - 295)
    cv2.rectangle(panel, (left, 12), (img_w - 12, 78), (18, 22, 30), -1)
    cv2.addWeighted(panel, 0.56, img, 0.44, 0, img)

    cv2.putText(
        img,
        f"Zone Setup {mode}",
        (left + 12, 34),
        UI_FONT,
        0.52,
        (248, 250, 252),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "Left add  Right undo  N switch",
        (left + 12, 54),
        UI_FONT,
        0.38,
        (203, 213, 225),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "R reset  S save  ESC quit",
        (left + 12, 70),
        UI_FONT,
        0.38,
        (148, 163, 184),
        1,
        cv2.LINE_AA,
    )

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
        draw_header(display, mode)

        # Existing zones
        if existing_zebra:
            style = EXISTING_ZONE_STYLES["ZEBRA"]
            draw_polygon(display, existing_zebra, style["color"], style["label"])
        if existing_buffer:
            style = EXISTING_ZONE_STYLES["BUFFER"]
            draw_polygon(display, existing_buffer, style["color"], style["label"])

        # New zones
        style = ZONE_STYLES["ZEBRA"]
        draw_polygon(display, zebra_points, style["color"], f"{style['label']} Drawing")
        style = ZONE_STYLES["BUFFER"]
        draw_polygon(display, buffer_points, style["color"], f"{style['label']} Drawing")

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
