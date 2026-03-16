def get_centroid(x1, y1, x2, y2):
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy

def get_bottom_center(x1, y1, x2, y2):
    cx = int((x1 + x2) / 2)
    cy = int(y2)  # Use the bottom edge
    return cx, cy
