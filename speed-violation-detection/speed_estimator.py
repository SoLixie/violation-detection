from utils import check_line_crossing


class SpeedEstimator:
    def __init__(self, fps, distance_meters=10):
        self.fps = fps
        self.distance_meters = distance_meters
        self.entry_frames = {}
        self.prev_positions = {}
        self.vehicle_speeds = {}

    def update(self, track_id, curr_point, frame_idx, line1, line2):
        prev_point = self.prev_positions.get(track_id)
        self.prev_positions[track_id] = curr_point

        if prev_point is None:
            return None

        if track_id not in self.entry_frames:
            if check_line_crossing(prev_point, curr_point, line1):
                self.entry_frames[track_id] = frame_idx

        if track_id in self.entry_frames and track_id not in self.vehicle_speeds:
            if check_line_crossing(prev_point, curr_point, line2):
                start = self.entry_frames[track_id]
                frames = frame_idx - start

                if frames > 0:
                    time = frames / self.fps
                    if time > 0.1:
                        speed = (self.distance_meters / time) * 3.6
                        if 0.1 <= speed <= 200:
                            self.vehicle_speeds[track_id] = speed

        return self.vehicle_speeds.get(track_id, None)