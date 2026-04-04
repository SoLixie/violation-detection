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


def update_tracker(detections, frame):
    h, w = frame.shape[:2]
    embeds = []

    for (x, y, bw, bh), conf, cls in detections:
        cx = x + bw / 2
        cy = y + bh / 2
        embeds.append(np.array([cx/w, cy/h, bw/w, bh/h], dtype=np.float32))

    tracks = tracker.update_tracks(detections, embeds=embeds)
    results = []

    for t in tracks:
        if not t.is_confirmed():
            continue
        l, t_, r, b = map(int, t.to_ltrb())
        results.append((t.track_id, l, t_, r, b, t.det_class))

    return results