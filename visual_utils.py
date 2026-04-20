import cv2
import numpy as np

# =========================================================
# UI CONFIGURATION & FONT
# =========================================================
UI_FONT = cv2.FONT_HERSHEY_DUPLEX
BG_PANEL_COLOR = (30, 24, 20)  # BGR: Dark Charcoal

# =========================================================
# COLOR SYSTEM (BGR - OpenCV Format)
# =========================================================
COLORS = {
    "normal": (255, 220, 200),         # Ice Blue
    "speed_detected": (0, 200, 0),     # Vibrant Green
    "speed": (255, 0, 0),               # True Blue
    "parking": (0, 191, 255),          # Amber/Orange
    "both": (255, 0, 255),             # Magenta
}

def get_color(vtype):
    return COLORS.get(vtype, COLORS["normal"])

# =========================================================
# BOLD LABEL RENDERER
# =========================================================
def draw_label(frame, text, origin, color, font_scale=0.5, align="center", solid=True):
    x, y = origin
    h, w = frame.shape[:2]
    
    # Thickness 2 creates the "Bold" look
    thickness = 2
    (tw, th), base = cv2.getTextSize(text, UI_FONT, font_scale, thickness)

    pad_x, pad_y = 10, 8
    box_w = tw + pad_x * 2
    box_h = th + base + pad_y * 2

    if align == "center":
        x = x - box_w // 2

    x1, y1 = max(5, x), max(5, y - box_h)
    x2, y2 = min(w - 5, x1 + box_w), min(h - 5, y)

    # Background
    if solid:
        cv2.rectangle(frame, (x1, y1), (x2, y2), BG_PANEL_COLOR, -1)
    else:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), BG_PANEL_COLOR, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Bold Border
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # White Bold Text
    cv2.putText(frame, text, (x1 + pad_x, y2 - pad_y),
                UI_FONT, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

# =========================================================
# VEHICLE BOX & RENDERER
# =========================================================
def draw_vehicle_box(frame, box, color, emphasis=False):
    x1, y1, x2, y2 = box
    if emphasis:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

def draw_vehicle_label(frame, bbox, track_id, speed, has_speed, vtype):
    x1, y1, x2, y2 = bbox
    color = get_color(vtype)

    # ✅ Always format speed properly
    if has_speed and speed is not None:
        speed_text = f"{int(speed)} km/h"
    else:
        speed_text = "0 km/h"

    label = f"ID:{track_id} | {speed_text}"

    draw_vehicle_box(frame, (x1, y1, x2, y2), color, emphasis=(vtype != "normal"))

    cx, cy = int((x1 + x2) / 2), y1 - 10
    draw_label(frame, label, (cx, cy), color, font_scale=0.45, solid=True)

# =========================================================
# REDESIGNED STATUS HUD (BOLD & ALIGNED)
# =========================================================
def draw_status_hud(frame, total_violations, info_text, speed_count=0, parking_count=0):
    frame_w = frame.shape[1]
    
    # HUD Panel Dimensions
    p_w, p_h = 340, 110
    px1, py1 = frame_w - p_w - 20, 20
    px2, py2 = frame_w - 20, 20 + p_h

    # Semi-transparent Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (px1, py1), (px2, py2), BG_PANEL_COLOR, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (px1, py1), (px2, py2), (200, 200, 200), 1, cv2.LINE_AA)

    # Left Section: Main Counter
    cv2.putText(frame, "VIOLATIONS", (px1 + 20, py1 + 35), UI_FONT, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, str(total_violations), (px1 + 20, py1 + 90), UI_FONT, 1.8, (255, 255, 255), 3, cv2.LINE_AA)

    # Vertical Divider
    cv2.line(frame, (px1 + 145, py1 + 20), (px1 + 145, py2 - 20), (100, 100, 100), 1)

    # Right Section: Detailed Stats
    col2_x = px1 + 165
    cv2.putText(frame, info_text.upper(), (col2_x, py1 + 35), UI_FONT, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(frame, f"SPEED: {speed_count}", (col2_x, py1 + 65), UI_FONT, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"PARKING: {parking_count}", (col2_x, py1 + 92), UI_FONT, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

# =========================================================
# ZONES & WINDOW
# =========================================================
def draw_speed_zones(frame, line1, line2):
    # Line 1 (Entry - Orange/Amber) | Line 2 (Exit - Blue)
    c1, c2 = (0, 165, 255), (255, 0, 0)
    cv2.line(frame, tuple(line1[0]), tuple(line1[1]), c1, 3)
    cv2.line(frame, tuple(line2[0]), tuple(line2[1]), c2, 3)

    for line, color, name in [(line1, c1, "ENTRY"), (line2, c2, "EXIT")]:
        for pt in [line[0], line[1]]:
            cv2.circle(frame, tuple(pt), 8, (255, 255, 255), -1)
            cv2.circle(frame, tuple(pt), 8, color, 3)
        # Bold Zone Labels
        draw_label(frame, name, (line[0][0], line[0][1] - 25), color, solid=True)

def draw_parking_zones(frame, parking_zone, buffer_zone):
    p_color = COLORS["parking"]
    b_color = (255, 255, 0) # Cyan
    
    if len(parking_zone) >= 3:
        cv2.polylines(frame, [np.array(parking_zone)], True, p_color, 3)
        moment = cv2.moments(np.array(parking_zone))
        if moment["m00"] != 0:
            cx, cy = int(moment["m10"]/moment["m00"]), int(moment["m01"]/moment["m00"])
            draw_label(frame, "PARKING ZONE", (cx, cy), p_color, solid=True)

    if len(buffer_zone) >= 3:
        cv2.polylines(frame, [np.array(buffer_zone)], True, b_color, 2)

def setup_display_window(name="Smart Zebra", w=1280, h=720):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, w, h)
    return name