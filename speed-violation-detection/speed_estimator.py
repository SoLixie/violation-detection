from utils import check_line_crossing


class SpeedEstimator:
    def __init__(self, fps, distance_meters=10):
        self.fps = fps
        self.distance_meters = distance_meters
        self.zone_entry_frames = {}
        self.prev_positions = {}
        self.vehicle_measurements = {}

    def update(self, track_id, curr_point, frame_idx, zone1, zone2, zone_type="line", vehicle_box=None):
        events = []
        prev_point = self.prev_positions.get(track_id)
        self.prev_positions[track_id] = curr_point

        if track_id not in self.zone_entry_frames:
            self.zone_entry_frames[track_id] = {}

        zone1_reached = self._zone_reached(prev_point, curr_point, zone1, zone_type, vehicle_box)
        zone2_reached = self._zone_reached(prev_point, curr_point, zone2, zone_type, vehicle_box)

        if zone1_reached and 1 not in self.zone_entry_frames[track_id]:
            self.zone_entry_frames[track_id][1] = frame_idx
            events.append("zone1")

        if zone2_reached and 2 not in self.zone_entry_frames[track_id]:
            self.zone_entry_frames[track_id][2] = frame_idx
            events.append("zone2")

        if track_id not in self.vehicle_measurements:
            entry_frames = self.zone_entry_frames[track_id]
            if 1 in entry_frames and 2 in entry_frames:
                frames = abs(entry_frames[2] - entry_frames[1])
                if frames > 0:
                    elapsed_seconds = frames / self.fps
                    if elapsed_seconds > 0.1:
                        speed = (self.distance_meters / elapsed_seconds) * 3.6
                        if speed > 0:
                            self.vehicle_measurements[track_id] = {
                                "speed_kmph": speed,
                                "frames": frames,
                                "time_seconds": elapsed_seconds,
                            }
                            events.append("measurement")

        return self.vehicle_measurements.get(track_id), events

    def _zone_reached(self, prev_point, curr_point, zone, zone_type, vehicle_box):
        if zone_type == "box":
            if prev_point is None:
                return False
            return self._cross_box_center_axis(prev_point, curr_point, zone)

        if prev_point is None:
            return False
        return check_line_crossing(prev_point, curr_point, zone)

    def _cross_box_center_axis(self, prev_point, curr_point, box):
        (x1, y1), (x2, y2) = box
        width = abs(x2 - x1)
        height = abs(y2 - y1)

        if height >= width:
            center_x = (x1 + x2) / 2
            return (
                (prev_point[0] < center_x <= curr_point[0]) or
                (prev_point[0] > center_x >= curr_point[0])
            )

        center_y = (y1 + y2) / 2
        return (
            (prev_point[1] < center_y <= curr_point[1]) or
            (prev_point[1] > center_y >= curr_point[1])
        )
