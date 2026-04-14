def get_centroid(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bottom_center(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int(y2)


def point_side_of_line(px, py, x1, y1, x2, y2):
    return (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)


def check_line_crossing(prev_point, curr_point, line):
    (x1, y1), (x2, y2) = line
    prev_side = point_side_of_line(prev_point[0], prev_point[1], x1, y1, x2, y2)
    curr_side = point_side_of_line(curr_point[0], curr_point[1], x1, y1, x2, y2)
    return prev_side * curr_side < 0
