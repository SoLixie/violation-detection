from core.geometry import check_line_crossing


class SpeedEstimator:
    def __init__(self, fps, distance, debug=False, timeout_seconds=10.0):
        self.fps = fps
        self.distance = distance
        self.debug = debug
        self.timeout_seconds = timeout_seconds
        self.data = {}

    def _create_track_state(self):
        return {
            "prev_point": None,
            "first_line": None,
            "first_time": None,
            "first_frame_index": None,
            "last_cross_time": None,
        }

    def _reset_measurement(self, track_state):
        track_state["first_line"] = None
        track_state["first_time"] = None
        track_state["first_frame_index"] = None
        track_state["last_cross_time"] = None

    def update(self, tid, point, timestamp_seconds, line1, line2, frame_index=None):
        if tid not in self.data:
            self.data[tid] = self._create_track_state()

        track_state = self.data[tid]

        if (
            track_state["first_time"] is not None
            and timestamp_seconds - track_state["first_time"] > self.timeout_seconds
        ):
            if self.debug:
                print(f"[DEBUG] Track {tid} speed window timed out; resetting")
            self._reset_measurement(track_state)

        prev_point = track_state["prev_point"]
        crossed_line1 = False
        crossed_line2 = False

        if prev_point is not None:
            crossed_line1 = check_line_crossing(prev_point, point, line1)
            crossed_line2 = check_line_crossing(prev_point, point, line2)

        track_state["prev_point"] = point

        if not crossed_line1 and not crossed_line2:
            return None

        crossed_lines = []
        if crossed_line1:
            crossed_lines.append(("line1", timestamp_seconds, frame_index))
        if crossed_line2:
            crossed_lines.append(("line2", timestamp_seconds, frame_index))

        for line_name, crossed_time, crossed_frame_index in crossed_lines:
            if self.debug:
                print(f"[DEBUG] Track {tid} crossed {line_name} at t={crossed_time:.3f}s")

            if track_state["first_line"] is None:
                track_state["first_line"] = line_name
                track_state["first_time"] = crossed_time
                track_state["first_frame_index"] = crossed_frame_index
                track_state["last_cross_time"] = crossed_time
                continue

            if line_name == track_state["first_line"]:
                track_state["last_cross_time"] = crossed_time
                continue

            elapsed_seconds = crossed_time - track_state["first_time"]

            if 0.1 < elapsed_seconds < 10.0:
                speed = (self.distance / elapsed_seconds) * 3.6
                if 0 < speed < 200:
                    first_frame_index = track_state["first_frame_index"]
                    second_frame_index = crossed_frame_index
                    result = {
                        "speed_kmph": speed,
                        "frames": (
                            abs(second_frame_index - first_frame_index)
                            if first_frame_index is not None and second_frame_index is not None
                            else None
                        ),
                        "time_seconds": elapsed_seconds,
                        "first_line": track_state["first_line"],
                        "second_line": line_name,
                        "crossed_line1_frame": (
                            first_frame_index
                            if track_state["first_line"] == "line1"
                            else second_frame_index
                        ),
                        "crossed_line2_frame": (
                            first_frame_index
                            if track_state["first_line"] == "line2"
                            else second_frame_index
                        ),
                    }
                    if self.debug:
                        print(
                            f"[DEBUG] Track {tid} speed computed: "
                            f"{speed:.1f} km/h over {elapsed_seconds:.2f}s"
                        )

                    self._reset_measurement(track_state)
                    return result

            if self.debug:
                print(f"[DEBUG] Track {tid} invalid speed window; re-arming from {line_name}")
            track_state["first_line"] = line_name
            track_state["first_time"] = crossed_time
            track_state["first_frame_index"] = crossed_frame_index
            track_state["last_cross_time"] = crossed_time

        return None

    def reset_track(self, tid):
        if self.debug:
            print(f"[DEBUG] Resetting speed state for track {tid}")
        self.data.pop(tid, None)

    def get_track_state(self, tid):
        return self.data.get(tid, {})
