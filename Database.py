import os
import cv2
import numpy as np
import shutil
from Signatures import *


PATH = "./database/signatures/"
FULL_VIDEOS_PATH = "./videos/"
CROPPED_VIDEOS_PATH = "./videos_cropped/"

def createDirectories(videos_folder, cropped_videos=False):
    videos = os.listdir(videos_folder)
    for video in videos:
        for sign_type in sign_types:
            if not os.path.exists(PATH + sign_type):
                os.makedirs(PATH + sign_type)

            if cropped_videos:
                spl = video.split("_from_")
                video_name = spl[0]
                segment_name = spl[1][:-5]
            else:
                video_name = video[:-4]
                segment_name = "full"

            data_path = PATH + sign_type + "/" + video_name
            if not os.path.exists(data_path):
                os.makedirs(data_path)

            video_path = videos_folder + video
            signature = sign_methods[sign_type](video_path)

            f = open(data_path + "/" + segment_name + ".txt", "wb")
            f.write(signature.tobytes())
            f.close()


full_signs = {}
cropped_signs = {}

def loadFullVideos():
    videos = os.listdir(FULL_VIDEOS_PATH)
    for video in videos:
        full_signs[video[:-4]] = {}
        for sign_method in sign_methods.keys():
            sign_file = PATH + sign_method + "/" + video[:-4] + "/full.txt"
            with open(sign_file, "rb") as f:
                compare_sign_bts = f.read()
                compare_sign = np.frombuffer(compare_sign_bts)
                full_signs[video[:-4]][sign_method] = compare_sign

def loadCroppedVideos():
    videos = os.listdir(CROPPED_VIDEOS_PATH)
    for video in videos:
        spl = video.split("_from_")
        video_name = spl[0]
        segment_name = spl[1][:-5]
        cropped_signs[video_name] = {}
        for sign_method in sign_methods.keys():
            sign_file = PATH + sign_method + "/" + video_name + "/" + segment_name + ".txt"
            cropped_signs[video_name][sign_method] = []
            with open(sign_file, "rb") as f:
                compare_sign_bts = f.read()
                compare_sign = np.frombuffer(compare_sign_bts)

                cropped_signs[video_name][sign_method].append(compare_sign)

def loadDatabase():
    loadFullVideos()
    loadCroppedVideos()


if __name__ == '__main__':
    loadDatabase()
    print(cropped_signs)

    refresh_database = False
    if refresh_database:
        # clean the previous database
        shutil.rmtree("database/signatures")
        os.makedirs("database/signatures")

        createDirectories(FULL_VIDEOS_PATH)
        createDirectories(CROPPED_VIDEOS_PATH, True)
