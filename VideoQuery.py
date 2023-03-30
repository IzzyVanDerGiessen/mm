import Database
import os
from Signatures import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

def query(video_path):
    """
        The brute force version of the query.
    """
    start = time.time()

    frames = get_video_frames(video_path)
    fps = get_fps(video_path)

    all_best = {}
    for sign_method_name in sign_methods[:2]:
        sign_method = sign_methods[sign_method_name]
        query_sign = sign_method(frames, False)

        results = {}
        videos = os.listdir(Database.FULL_VIDEOS_PATH)
        for video in videos:
            print(video)
            y = []
            test_len = vid_len(Database.FULL_VIDEOS_PATH + video)
            test_fps = get_fps(Database.FULL_VIDEOS_PATH + video)

            if test_fps == 0:
                continue

            # step size in frames
            step_size = int((len(frames) / fps) * test_fps)

            cap = cv2.VideoCapture(Database.FULL_VIDEOS_PATH + video)
            for i in range(0, test_len-len(frames), step_size):
                #print(str(i / step_size) + "/" + str(int((test_len-len(frames))/step_size)))
                sample = []
                for j in range(i, i+len(frames)):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    sample.append(frame)

                test_sign = sign_method(sample, False)
                try:
                    score = np.abs(test_sign - query_sign).sum()
                except:
                    continue
                if score == score:
                    y.append((video + ": " + str(i/test_fps) + "-" + str((i+step_size)/test_fps), score))
                else:
                    print(score)

            results[video] = sorted(y, key=lambda x: x[1])

        best = []
        for res in results.keys():
            best += results[res][:5]
        all_best[sign_method_name] = best

    for best in all_best.keys():
        print(best)
        print(all_best[best])
        print("\n")
    print(time.time() - start)

def vid_len(video):
    cap = cv2.VideoCapture(video)
    vid_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return vid_len

def get_fps(video):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def euclidean_norm_mean(x,y):
    """
        Copied from the lab
    """
    x = np.mean(x, axis=0)
    y = np.mean(y, axis=0)
    return np.linalg.norm(x-y)

def colorhist2(frames):
    """
        Copied from the lap (very slow!)
    """
    hists = []
    for frame in frames:
        try:
            chans = cv2.split(frame)
        except:
            print(frame)
            x = 1/0
            print(x)
        color_hist = np.zeros((256,len(chans)))
        for i in range(len(chans)):
            color_hist[:,i] = np.histogram(chans[i], bins=np.arange(256+1))[0]/float((chans[i].shape[0]*chans[i].shape[1]))
        hists.append(color_hist)
    return hists

def colorhist(frames):
    avg_hists = np.zeros(256)
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = np.bincount(gray.flatten(), None, 256)
        avg_hists += hist

    return avg_hists / len(frames)

query("./videos_cropped/British_Plugs_Are_Better_from_0.0_to_5.0).mp4")
