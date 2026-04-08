import cv2
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "speed_config.json"
VIDEO_PATH = PROJECT_ROOT / "vids" / "vid1.mp4"
WINDOW_NAME = "Draw Boxes"
MAX_BOXES = 2

boxes = []
draft_start = None
draft_box = None
is_drawing = False


def normalize_box(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    return [(min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2))]


def save_boxes():
    if len(boxes) != MAX_BOXES:
        print("Need 2 boxes before saving!")
        return

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    config = {}
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            config = json.load(f)

    config["box1"] = boxes[0]
    config["box2"] = boxes[1]

    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    print("Boxes saved!")


def mouse_callback(event, x, y, flags, param):
    global draft_start, draft_box, is_drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(boxes) >= MAX_BOXES:
            return
        draft_start = (x, y)
        draft_box = [draft_start, draft_start]
        is_drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and is_drawing and draft_start is not None:
        draft_box = normalize_box(draft_start, (x, y))
    elif event == cv2.EVENT_LBUTTONUP and is_drawing and draft_start is not None:
        draft_box = normalize_box(draft_start, (x, y))
        is_drawing = False


def main():
    global draft_start, draft_box, is_drawing

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    ret, frame = cap.read()
    if not ret:
        print("Error loading video")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print("Draw a box with click-and-drag.")
    print("Press 'n' to confirm the current box.")
    print("Press 'u' to remove the last box.")
    print("Press 's' to save 2 boxes.")

    while True:
        temp = frame.copy()

        for idx, box in enumerate(boxes, start=1):
            cv2.rectangle(temp, box[0], box[1], (0, 0, 255), 2)
            cv2.putText(temp, f"Box {idx}", (box[0][0], max(30, box[0][1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if draft_box is not None:
            cv2.rectangle(temp, draft_box[0], draft_box[1], (0, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, temp)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key in (ord("n"), ord("N")):
            if draft_box is None:
                print("Draw a box first, then press 'n'.")
            elif len(boxes) >= MAX_BOXES:
                print("Maximum number of boxes is 2.")
            else:
                boxes.append(draft_box)
                print(f"Box {len(boxes)} confirmed: {draft_box}")
                draft_start = None
                draft_box = None
                is_drawing = False
        elif key in (ord("u"), ord("U")):
            if boxes:
                removed = boxes.pop()
                print(f"Removed box {len(boxes) + 1}: {removed}")
            else:
                print("No confirmed boxes to remove.")
        elif key in (ord("r"), ord("R")):
            draft_start = None
            draft_box = None
            is_drawing = False
        elif key in (ord("s"), ord("S")):
            if len(boxes) != MAX_BOXES:
                print("Confirm 2 boxes before saving.")
            else:
                save_boxes()
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
