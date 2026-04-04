from utils import check_line_crossing

class SpeedEstimator:
    def __init__(self, fps, distance_meters=10):
        self.fps = fps
        self.distance_meters = distance_meters
        self.entry_frames = {}      # {track_id: frame_idx}
        self.prev_positions = {}    # {track_id: (x, y)}
        self.vehicle_speeds = {}    # {track_id: speed_kmph}

        if fps <= 0:
            raise ValueError("FPS must be positive")
        if distance_meters <= 0:
            raise ValueError("Distance must be positive")

    def update(self, track_id, curr_point, frame_idx, line1, line2):
        """
        curr_point: (x, y) → use get_bottom_center()
        line1, line2: ((x1, y1), (x2, y2))
        """

        if not isinstance(frame_idx, int):
            return None

        prev_point = self.prev_positions.get(track_id)
        self.prev_positions[track_id] = curr_point

        if prev_point is None:
            return None

        # 1️⃣ Entry detection (cross first line)
        if track_id not in self.entry_frames:
            if check_line_crossing(prev_point, curr_point, line1):
                self.entry_frames[track_id] = frame_idx

        # 2️⃣ Speed calculation (cross second line)
        if track_id in self.entry_frames and track_id not in self.vehicle_speeds:
            if check_line_crossing(prev_point, curr_point, line2):

                start_frame = self.entry_frames[track_id]
                frames_elapsed = frame_idx - start_frame

                if frames_elapsed > 0:
                    time_elapsed = frames_elapsed / self.fps

                    if time_elapsed > 0.1:  # avoid noise
                        speed_mps = self.distance_meters / time_elapsed
                        speed_kmph = speed_mps * 3.6

                        if 0.1 <= speed_kmph <= 200:
                            self.vehicle_speeds[track_id] = speed_kmph

        return self.vehicle_speeds.get(track_id, None)