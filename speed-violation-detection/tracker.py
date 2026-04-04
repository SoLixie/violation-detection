import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

tracker = DeepSort(
    max_iou_distance=0.7,
    max_age=30,
    n_init=3,
    max_cosine_distance=0.4,
    nn_budget=20,
    gating_only_position=True,
    embedder=None,
)

# Utility: check which side of line a point is on
def point_side_of_line(px, py, x1, y1, x2, y2):
    return (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)

def update_tracker(detections, frame, line1=None, line2=None):
    """
    line1, line2 format:
    ((x1, y1), (x2, y2))
    Example:
    line1 = ((100, 200), (500, 200))
    """

    frame_height, frame_width = frame.shape[:2]
    embeds = []

    for (x, y, w, h), confidence, class_id in detections:
        cx = x + (w / 2.0)
        cy = y + (h / 2.0)
        area = max(w * h, 1)
        aspect_ratio = w / max(h, 1)

        embeds.append(
            np.array(
                [
                    cx / max(frame_width, 1),
                    cy / max(frame_height, 1),
                    w / max(frame_width, 1),
                    h / max(frame_height, 1),
                    min(aspect_ratio, 10.0) / 10.0,
                    area / max(frame_width * frame_height, 1),
                    float(confidence),
                    float(class_id) / 10.0,
                ],
                dtype=np.float32,
            )
        )

    tracks = tracker.update_tracks(detections, embeds=embeds)

    results = []

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        class_id = track.det_class

        l, t, r, b = map(int, track.to_ltrb())

        # Center of tracked box
        cx = int((l + r) / 2)
        cy = int((t + b) / 2)

        # Check line crossing (if lines provided)
        crossed_line1 = False
        crossed_line2 = False

        if line1:
            (x1, y1), (x2, y2) = line1
            side = point_side_of_line(cx, cy, x1, y1, x2, y2)
            crossed_line1 = side > 0

        if line2:
            (x1, y1), (x2, y2) = line2
            side = point_side_of_line(cx, cy, x1, y1, x2, y2)
            crossed_line2 = side > 0

        results.append(
            (track_id, l, t, r, b, class_id, crossed_line1, crossed_line2)
        )

    return results