import cv2
import json
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "speed_config.json"
WINDOW_NAME = "Draw Lines"
MAX_LINES = 2
LINE_STYLES = (
    {"name": "Entry", "color": (64, 92, 255)},
    {"name": "Exit", "color": (255, 191, 64)},
)
DRAWING_STYLE = {"name": "Draft", "color": (64, 224, 208)}

lines = []
draft_start = None
draft_line = None
is_drawing = False
moving_point = None  # (line_index, point_index)


def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_video_path(config):
    video_value = config.get("video_path")
    if not video_value:
        raise KeyError("speed_config.json must define video_path")

    video_path = Path(video_value)
    if video_path.is_absolute():
        return video_path

    candidate = (PROJECT_ROOT / video_path).resolve()
    if candidate.exists():
        return candidate

    if video_path.parts and video_path.parts[0] == "vids":
        legacy_candidate = PROJECT_ROOT / "videos" / Path(*video_path.parts[1:])
        if legacy_candidate.exists():
            return legacy_candidate.resolve()

    raise FileNotFoundError(f"Could not find video file: {candidate}")


def save_lines():
    if len(lines) != MAX_LINES:
        print("Need 2 lines before saving!")
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


def near_point(p, q, threshold=10):
    return np.linalg.norm(np.array(p) - np.array(q)) < threshold


def mouse_callback(event, x, y, flags, param):
    global draft_start, draft_line, is_drawing, moving_point

    if event == cv2.EVENT_LBUTTONDOWN:

        # Check if clicking near existing endpoints → adjust
        for i, line in enumerate(lines):
            if near_point((x, y), line[0]):
                moving_point = (i, 0)
                return
            elif near_point((x, y), line[1]):
                moving_point = (i, 1)
                return

        # Otherwise start drawing
        if len(lines) >= MAX_LINES:
            return

        draft_start = (x, y)
        draft_line = [draft_start, draft_start]
        is_drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:

        if moving_point is not None:
            line_idx, pt_idx = moving_point
            lines[line_idx][pt_idx] = (x, y)

        elif is_drawing and draft_start is not None:
            draft_line = [draft_start, (x, y)]

    elif event == cv2.EVENT_LBUTTONUP:

        if moving_point is not None:
            moving_point = None
            return

        if is_drawing and draft_start is not None:
            draft_line = [draft_start, (x, y)]
            is_drawing = False


def draw_line(img, line, color, label):
    pt1, pt2 = line

    overlay = img.copy()
    cv2.line(overlay, pt1, pt2, color, 10, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.16, img, 0.84, 0, img)

    cv2.line(img, pt1, pt2, (255, 255, 255), 6, cv2.LINE_AA)
    cv2.line(img, pt1, pt2, color, 3, cv2.LINE_AA)

    cv2.circle(img, pt1, 10, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, pt2, 10, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, pt1, 6, color, -1, cv2.LINE_AA)
    cv2.circle(img, pt2, 6, color, -1, cv2.LINE_AA)

    anchor_x = int((pt1[0] + pt2[0]) / 2)
    anchor_y = max(36, int((pt1[1] + pt2[1]) / 2) - 18)
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
    )
    left = max(8, anchor_x - text_width // 2 - 10)
    top = max(8, anchor_y - text_height - baseline - 10)
    right = left + text_width + 20
    bottom = top + text_height + baseline + 14

    panel = img.copy()
    cv2.rectangle(panel, (left, top), (right, bottom), (20, 24, 32), -1)
    cv2.rectangle(panel, (left, top), (right, bottom), color, 2)
    cv2.addWeighted(panel, 0.72, img, 0.28, 0, img)

    cv2.putText(
        img,
        label,
        (left + 10, bottom - baseline - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (245, 247, 250),
        2,
        cv2.LINE_AA,
    )


def draw_header(img):
    panel = img.copy()
    cv2.rectangle(panel, (16, 16), (540, 118), (18, 22, 30), -1)
    cv2.addWeighted(panel, 0.72, img, 0.28, 0, img)

    cv2.putText(
        img,
        "Speed Line Setup",
        (32, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (248, 250, 252),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "Drag to draw, N confirm, U undo, S save, ESC exit",
        (32, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (203, 213, 225),
        1,
        cv2.LINE_AA,
    )


def main():
    global draft_start, draft_line, is_drawing

    config = load_config()
    video_path = resolve_video_path(config)

    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    if not ret:
        print(f"Error loading video: {video_path}")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print("Draw a line with click-and-drag.")
    print("Press 'n' to confirm the current line.")
    print("Click endpoints to adjust.")
    print("Press 'u' to remove the last line.")
    print("Press 's' to save 2 lines.")

    while True:
        temp = frame.copy()
        draw_header(temp)

        for idx, line in enumerate(lines, start=1):
            style = LINE_STYLES[min(idx - 1, len(LINE_STYLES) - 1)]
            draw_line(temp, line, style["color"], f"{style['name']} Line")

        if draft_line is not None:
            draw_line(temp, draft_line, DRAWING_STYLE["color"], DRAWING_STYLE["name"])

        cv2.imshow(WINDOW_NAME, temp)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        elif key in (ord("n"), ord("N")):
            if draft_line is None:
                print("Draw a line first, then press 'n'.")
            elif len(lines) >= MAX_LINES:
                print("Maximum number of lines is 2.")
            else:
                lines.append(draft_line)
                print(f"Line {len(lines)} confirmed: {draft_line}")
                draft_start = None
                draft_line = None
                is_drawing = False

        elif key in (ord("u"), ord("U")):
            if lines:
                removed = lines.pop()
                print(f"Removed line: {removed}")
            else:
                print("No lines to remove.")

        elif key in (ord("r"), ord("R")):
            draft_start = None
            draft_line = None
            is_drawing = False

        elif key in (ord("s"), ord("S")):
            if len(lines) != MAX_LINES:
                print("Confirm 2 lines before saving.")
            else:
                save_lines()
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

