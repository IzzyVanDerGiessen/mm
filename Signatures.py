import cv2
import numpy as np

sign_types = ["colorhists"]

def signColorhists(video_path):
    print(video_path)
    cap = cv2.VideoCapture(video_path)
    ret = True
    fps = cap.get(cv2.CAP_PROP_FPS)
    i = 0
    avg_hists = np.zeros(256)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = np.bincount(gray.flatten(), None, 256)
        avg_hists += hist
        i += 1
    avg_hists /= i
    return avg_hists


# the keys correspond to the folder names
sign_methods = {
    "colorhists": signColorhists
}
