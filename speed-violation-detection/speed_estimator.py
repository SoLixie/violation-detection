class SpeedEstimator:
    def __init__(self, fps, distance_meters=10):
        self.fps = fps
        self.distance_meters = distance_meters
        self.entry_frames = {}      # {track_id: frame_idx}
        self.prev_positions = {}    # {track_id: last_cy}
        self.vehicle_speeds = {}    # {track_id: speed_kmph}

    def update(self, track_id, cy, frame_idx, line1, line2):
        prev_cy = self.prev_positions.get(track_id)
        self.prev_positions[track_id] = cy

        if prev_cy is None:
            return None

        # Determine which line is "entry" based on movement direction
        # Movement Down: Line 1 (Upper) -> Line 2 (Lower)
        # Movement Up: Line 2 (Lower) -> Line 1 (Upper)

        # 1. Entry detection
        # Crossing Line 1 moving Down OR Crossing Line 2 moving Up
        if (prev_cy <= line1 < cy) or (prev_cy >= line2 > cy):
            if track_id not in self.entry_frames:
                self.entry_frames[track_id] = frame_idx

        # 2. Speed calculation
        # Crossing Line 2 moving Down OR Crossing Line 1 moving Up
        if (prev_cy <= line2 < cy) or (prev_cy >= line1 > cy):
            if track_id in self.entry_frames and track_id not in self.vehicle_speeds:
                start_frame = self.entry_frames[track_id]
                frames_elapsed = frame_idx - start_frame
                
                if frames_elapsed > 0:
                    time_elapsed = frames_elapsed / self.fps
                    speed_mps = self.distance_meters / time_elapsed
                    speed_kmph = speed_mps * 3.6
                    self.vehicle_speeds[track_id] = speed_kmph

        return self.vehicle_speeds.get(track_id, None)

