import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

# SHARED TRACKER - NO DUPLICATION
tracker = DeepSort(
    max_age=30,
    n_init=3,
    max_cosine_distance=0.4,
    embedder=None  # IOU-only for speed
)

def update_tracker(detections, frame):
    """
    INPUT: detections = [([x, y, w, h], confidence, class_id), ...]
    OUTPUT: tracks = [(track_id, x1, y1, x2, y2, class_id), ...]
    UNIFIED FORMAT for main_detection.py integration
    """
    dummy_embeds = [np.ones(8, dtype=np.float32) for _ in detections]
    tracks = tracker.update_tracks(detections, embeds=dummy_embeds)
    results = []

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        class_id = track.det_class
        l, t, r, b = map(int, track.to_ltrb())

        # Filter tiny detections
        if (r - l) < 20 or (b - t) < 20:
            continue

        results.append((track_id, l, t, r, b, class_id))

    return results

