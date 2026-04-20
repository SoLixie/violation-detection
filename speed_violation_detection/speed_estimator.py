from common.geometry import check_line_crossing

class SpeedEstimator:
    def __init__(self, fps, distance, debug=False):
        self.fps = fps
        self.distance = distance
        self.data = {}  # Per-track isolated state
        self.debug = debug

    def update(self, tid, point, frame, line1, line2):
        """Update speed estimation for a single track.
        
        Args:
            tid: Track ID
            point: Current (x, y) position
            frame: Current frame index
            line1: First speed line [(x1,y1), (x2,y2)]
            line2: Second speed line [(x1,y1), (x2,y2)]
        
        Returns:
            Speed data dict if speed computed, None otherwise
        """
        if tid not in self.data:
            self.data[tid] = {
                "prev_point": None,
                "t1": None,
                "t2": None,
                "crossed_line1": False,
                "crossed_line2": False
            }

        d = self.data[tid]

        if d["prev_point"] is not None:
            # Check for line1 crossing (only if not already crossed)
            if not d["crossed_line1"] and check_line_crossing(d["prev_point"], point, line1):
                d["t1"] = frame
                d["crossed_line1"] = True
                if self.debug:
                    print(f"[DEBUG] Track {tid} crossed line1 at frame {frame}")
            
            # Check for line2 crossing (only if line1 already crossed and not already crossed line2)
            elif d["crossed_line1"] and not d["crossed_line2"] and check_line_crossing(d["prev_point"], point, line2):
                d["t2"] = frame
                d["crossed_line2"] = True
                if self.debug:
                    print(f"[DEBUG] Track {tid} crossed line2 at frame {frame}")

        d["prev_point"] = point

        # Compute speed if both lines crossed
        if d["t1"] is not None and d["t2"] is not None:
            elapsed_seconds = (d["t2"] - d["t1"]) / self.fps
            
            # Safety checks
            if elapsed_seconds > 0.1 and elapsed_seconds < 10.0:  # Reasonable time bounds
                speed = (self.distance / elapsed_seconds) * 3.6
                
                # Speed sanity check (max reasonable speed, e.g., 200 km/h)
                if 0 < speed < 200:
                    result = {
                        "speed_kmph": speed,
                        "frames": abs(d["t2"] - d["t1"]),
                        "time_seconds": elapsed_seconds,
                        "crossed_line1_frame": d["t1"],
                        "crossed_line2_frame": d["t2"]
                    }
                    if self.debug:
                        print(f"[DEBUG] Track {tid} speed computed: {speed:.1f} km/h over {elapsed_seconds:.2f}s")
                    
                    # Reset for next measurement (allow multiple speed measurements per track)
                    d["t1"] = None
                    d["t2"] = None
                    d["crossed_line1"] = False
                    d["crossed_line2"] = False
                    
                    return result

        return None

    def reset_track(self, tid):
        """Clear stored speed state for a track that has disappeared."""
        if self.debug:
            print(f"[DEBUG] Resetting speed state for track {tid}")
        self.data.pop(tid, None)
    
    def get_track_state(self, tid):
        """Get current state for debugging/visualization"""
        return self.data.get(tid, {})

