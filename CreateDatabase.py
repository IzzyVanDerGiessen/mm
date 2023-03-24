import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

database_path = "database/signatures/"

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
    #plt.plot(avg_hists)
    #plt.title(str(i / fps))
    #plt.show()
    return avg_hists

sign_methods = {
    "colorhists": signColorhists
}

def createDirectories(videos_folder):
    videos = os.listdir(videos_folder)
    sign_types = os.listdir(database_path)
    for sign_type in sign_types:
        for video in videos:
            data_path = database_path + sign_type + "/" + video[:-4]
            if not os.path.exists(data_path):
                os.makedirs(data_path)

            video_path = videos_folder + video
            signature = sign_methods[sign_type](video_path)

            f = open(data_path + "/full.txt", "w")
            f.write(np.array_str(signature))
            f.close()
            #x = 1/0
            #print(x)




createDirectories("./videos/")
