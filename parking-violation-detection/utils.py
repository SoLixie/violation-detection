def get_centroid(x1, y1, x2, y2):
    """
    Returns center of bounding box
    (used for general tracking/visualization)
    """
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy


def get_bottom_center(x1, y1, x2, y2):
    """
    Returns bottom-center of bounding box
    (MOST IMPORTANT for parking detection)
    """
    cx = int((x1 + x2) / 2)
    cy = int(y2)
    return cx, cy