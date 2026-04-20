import cv2
import json
import numpy as np

points = []
mode = "zebra"
config = {"zebra_zone": [], "buffer_zone": []}

def mouse(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x,y))

def main(video):
    global mode, points

    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()

    cv2.namedWindow("zone")
    cv2.setMouseCallback("zone", mouse)

    while True:
        temp = frame.copy()

        for p in points:
            cv2.circle(temp, p, 5, (0,255,0), -1)

        if len(points) > 2:
            cv2.polylines(temp, [np.array(points)], True, (0,255,0), 2)

        cv2.imshow("zone", temp)
        key = cv2.waitKey(1)

        if key == ord('s'):
            config["zebra_zone"] = points
            with open("config/parking_config.json","w") as f:
                json.dump(config,f,indent=4)
            break

        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()