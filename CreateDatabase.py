import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil

database_path = "database/signatures/"
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

def createDirectories(videos_folder, cropped_videos=False):
    # clean the previous database
    #shutil.rmtree("database/signatures")
    #os.makedirs("database/signatures")

    videos = os.listdir(videos_folder)
    for video in videos:
        for sign_type in sign_types:
            if not os.path.exists(database_path + sign_type):
                os.makedirs(database_path + sign_type)

            if cropped_videos:
                spl = video.split("_from_")
                video_name = spl[0]
                segment_name = spl[1][:-5]
            else:
                video_name = video[:-4]
                segment_name = "full"

            data_path = database_path + sign_type + "/" + video_name
            if not os.path.exists(data_path):
                os.makedirs(data_path)

            video_path = videos_folder + video
            signature = sign_methods[sign_type](video_path)

            f = open(data_path + "/" + segment_name + ".txt", "w")
            f.write(np.array_str(signature))
            f.close()


createDirectories("./videos/")
createDirectories("./videos_cropped/", True)
