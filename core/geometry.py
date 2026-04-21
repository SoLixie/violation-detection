import numpy as np
import cv2

def get_centroid(x1, y1, x2, y2):
    """Returns center of bounding box"""
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy

def get_bottom_center(x1, y1, x2, y2):
    """Returns bottom-center of boun    python main_detection.py --video path\to\video.mp4 --debug    python main_detection.py --video path\to\video.mp4 --debugding box (CRITICAL for ground-level tracking)"""
    cx = int((x1 + x2) / 2)
    cy = int(y2)
    return cx, cy

def point_side_of_line(px, py, x1, y1, x2, y2):
    """Determine which side of line point (px, py) lies"""
    return (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)

def check_line_crossing(prev_point, curr_point, line):
    """Check if the line segment from prev_point to curr_point intersects the line.
    
    Uses segment intersection algorithm to detect crossings that may occur
    between frames (handles fast-moving vehicles).
    """
    (x1, y1), (x2, y2) = line
    px1, py1 = prev_point
    px2, py2 = curr_point
    
    # Line segment intersection algorithm
    # Check if segment (px1,py1)-(px2,py2) intersects (x1,y1)-(x2,y2)
    
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0  # colinear
        return 1 if val > 0 else 2  # clock or counterclock wise
    
    def on_segment(p, q, r):
        if (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
            min(p[1], r[1]) <= q[1] <= max(p[1], r[1])):
            return True
        return False
    
    p1 = (x1, y1)
    q1 = (x2, y2)
    p2 = (px1, py1)
    q2 = (px2, py2)
    
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    
    if o1 != o2 and o3 != o4:
        return True
    
    # Special cases for colinear points
    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True
    
    return False

def is_inside_polygon(cx, cy, polygon):
    """Check if point (cx, cy) is inside polygon"""
    return cv2.pointPolygonTest(polygon, (float(cx), float(cy)), False) >= 0

def is_stationary(positions, threshold=5, min_samples=5):
    """Check if vehicle is stationary based on position history"""
    if len(positions) < min_samples:
        return False

    distances = [
        np.linalg.norm(np.array(positions[i]) - np.array(positions[i-1]))
        for i in range(1, len(positions))
    ]
    return np.mean(distances) < threshold

