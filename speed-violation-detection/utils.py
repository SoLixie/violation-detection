def get_centroid(x1, y1, x2, y2):
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy


def get_bottom_center(x1, y1, x2, y2):
    cx = int((x1 + x2) / 2)
    cy = int(y2)  # bottom edge
    return cx, cy


# NEW: Check which side of a line a point lies on
def point_side_of_line(px, py, x1, y1, x2, y2):
    return (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)


# NEW: Check if object crossed a line (basic version)
def check_line_crossing(prev_point, curr_point, line):
    """
    prev_point: (x, y) from previous frame
    curr_point: (x, y) from current frame
    line: ((x1, y1), (x2, y2))
    """

    (x1, y1), (x2, y2) = line

    prev_side = point_side_of_line(prev_point[0], prev_point[1], x1, y1, x2, y2)
    curr_side = point_side_of_line(curr_point[0], curr_point[1], x1, y1, x2, y2)

    # If signs are different → crossed the line
    return prev_side * curr_side < 0