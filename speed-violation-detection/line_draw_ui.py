import cv2
import json
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "speed_config.json"
VIDEO_PATH = PROJECT_ROOT / "videos" / "crossing1.mp4"

lines = []
current_points = []


def mouse_callback(event, x, y, flags, param):
    global current_points, lines

    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))

        if len(current_points) == 2:
            lines.append(current_points)
            current_points = []
            print(f"Line {len(lines)}: {lines[-1]}")


def main():
    global lines, current_points

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    ret, frame = cap.read()
    if not ret:
        print("Error loading video")
        return

    cv2.namedWindow("Draw Lines")
    cv2.setMouseCallback("Draw Lines", mouse_callback)

    while True:
        temp = frame.copy()

        for pt in current_points:
            cv2.circle(temp, pt, 5, (0, 255, 0), -1)

        for line in lines:
            cv2.line(temp, line[0], line[1], (0, 0, 255), 2)

        cv2.imshow("Draw Lines", temp)

        if cv2.waitKey(1) == 27 or len(lines) == 2:
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(lines) != 2:
        print("Need 2 lines!")
        return

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    config = {}
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            config = json.load(f)

    config["line1"] = lines[0]
    config["line2"] = lines[1]

    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    print("Lines saved!")


if __name__ == "__main__":
    main()
