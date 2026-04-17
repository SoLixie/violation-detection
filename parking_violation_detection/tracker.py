import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize tracker
tracker = DeepSort(
    max_age=30,              # how long to keep lost tracks
    n_init=3,                # frames needed to confirm a track
    max_cosine_distance=0.4, # appearance similarity threshold
    embedder=None            # use IOU-only tracking to avoid heavy Torch embedding
)

def update_tracker(detections, frame):
    """
    detections: [([x, y, w, h], confidence, class_id), ...]
    returns: [(track_id, x1, y1, x2, y2, class_id), ...]
    """

    dummy_embeds = [np.ones(8, dtype=np.float32) for _ in detections]
    tracks = tracker.update_tracks(detections, embeds=dummy_embeds)
    results = []

    for track in tracks:

        # Skip unconfirmed tracks (reduces noise)
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        class_id = track.det_class

        # Bounding box
        l, t, r, b = map(int, track.to_ltrb())

        # OPTIONAL: filter tiny boxes (reduces false detections)
        if (r - l) < 20 or (b - t) < 20:
            continue

        results.append((track_id, l, t, r, b, class_id))

    return results
