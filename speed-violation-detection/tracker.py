from deep_sort_realtime.deepsort_tracker import DeepSort

tracker = DeepSort(
    max_age=30,
    n_init=3,
    max_cosine_distance=0.4
)

def update_tracker(detections, frame):

    tracks = tracker.update_tracks(detections, frame=frame)

    results = []

    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id
        class_id = track.det_class

        l, t, r, b = map(int, track.to_ltrb())

        results.append((track_id, l, t, r, b, class_id))

    return results