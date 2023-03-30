import Database
import os
from Signatures import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import time


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type = str, default = "./videos_cropped/British_Plugs_Are_Better_from_0.0_to_5.0).mp4")
    parser.add_argument("--pipeline", type = str, default = "brute_force")
    parser.add_argument("--feature", type = str, default = "colorhists")
    return parser.parse_args()

def brute_force_pipeline(video_path, feature):
    """
        The brute force version of the query.
    """
    start = time.time()

    frames = getVideoFrames(video_path)
    fps = get_fps(video_path)

    query_feature = compute_feature(feature, path = video_path)

    results = {}
    videos = os.listdir(Database.FULL_VIDEOS_PATH)
    for video in videos:
        print(video)
        y = []
        test_len = vid_len(Database.FULL_VIDEOS_PATH + video)
        test_fps = get_fps(Database.FULL_VIDEOS_PATH + video)
        

        if test_fps == 0:
            continue

        test_samplerate, test_samples = wav.read(video.split('.mp4')[0] + ".wav")
        num_samples_per_frame = len(test_samples)//len(test_len)

        # step size in frames
        step_size = int((len(frames) / fps) * test_fps)

        cap = cv2.VideoCapture(Database.FULL_VIDEOS_PATH + video)
        for i in range(0, test_len-len(frames), step_size):

            sample = None
            # to extend the implementation for both video and audio
            if feature in  ["colorhists", "temporal_diff"]:
                sample = []
                for j in range(i, i+len(frames)):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    sample.append(frame)
            elif feature in ["mfccs", "audio_powers"]: 
                sample = test_samples[i:i+num_samples_per_frame]
                

            test_feature = compute_feature(feature, data = sample, samplerate= test_samplerate, num_frames = step_size)
            lengthLimiter = min(len(test_feature), len(query_feature)) #since sometimes we end up with weird numbers of frames (cuz of end of vid?)
            score = np.abs(test_feature[:lengthLimiter] - query_feature[:lengthLimiter]).sum()
            if score == score:
                y.append((video + ": " + str(i/test_fps) + "-" + str((i+step_size)/test_fps), score))
            else:
                print(score)

        results[video] = sorted(y, key=lambda x: x[1])

    best = []
    for res in results.keys():
        best += results[res][:5]
    print(sorted(best, key=lambda x: x[1])[:5])
    print(time.time() - start)

def signature_pipeline(file_path, feature):
    return None 



def query(video_path):
    """
        The brute force version of the query.
    """
    start = time.time()

    frames = getVideoFrames(video_path)
    fps = get_fps(video_path)

    query_colorhist = colorhist(frames)

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

            test_colorhist = colorhist(sample)
            score = np.abs(test_colorhist - query_colorhist).sum()
            if score == score:
                y.append((video + ": " + str(i/test_fps) + "-" + str((i+step_size)/test_fps), score))
            else:
                print(score)

        results[video] = sorted(y, key=lambda x: x[1])

    best = []
    for res in results.keys():
        best += results[res][:5]
    print(sorted(best, key=lambda x: x[1])[:5])
    print(time.time() - start)

def euclidean_norm_mean(x,y):
    """
        Copied from the lab
    """
    x = np.mean(x, axis=0)
    y = np.mean(y, axis=0)
    return np.linalg.norm(x-y)

def getVideoFrames(video_path):
    start = time.time()
    cap = cv2.VideoCapture(video_path)
    ret = True
    frames = []
    while ret:
        ret, frame = cap.read()
        frames.append(frame)
    cap.release()

    # remove the None at the end
    frames.pop()
    return frames

def compute_feature(feature, path = None, data = None, samplerate = None, num_frames = None):
    match feature:
        case "colorhists" : 
            if path != None:
                frames = getVideoFrames(path)
            if data != None:
                frames = data
            return sign_methods[feature](frames)
        case "mfccs":
            if path != None: 
                audio, sample_rate = librosa.load(path.split('.mp4')[0] + ".wav")
                if 'BlackKnight' in path:
                    audio, sample_rate = librosa.load(path.split('.avi')[0] + ".wav")
                signature = sign_methods[feature](audio, sample_rate)
            return None
        case "audio_powers":
            if path != None: 
                frames = getVideoFrames(path)
                samplerate, samples = wav.read(path.split('.mp4')[0] + ".wav")
                if 'BlackKnight' in path:
                    samplerate, samples = wav.read(path.split('.avi')[0] + ".wav")
                return sign_methods[feature](samplerate, samples, len(frames))
            if data != None:
                return sign_methods[feature](samplerate, data, num_frames)
                    
            return None # need to figure out how to do audio data
        
        case "temporal_diff":
            if path != None:
                frames = getVideoFrames(path)
                
            if data != None:
                frames = data
            return sign_methods[feature](frames)
        case _ :
            print("==================\n An error in the matching occured :) \n ==================")
            return -1 



if __name__ == '__main__':
    args = get_args()
    file_path = args.filepath
    pipeline = args.pipeline
    feature = args.feature


    if pipeline == "brute_force":
        brute_force_pipeline(file_path, feature)
    elif pipeline == "signatures":
        signature_pipeline(file_path, feature)

    

#query("./videos_cropped/British_Plugs_Are_Better_from_0.0_to_5.0).mp4")

# def colorhist2(frames):
#     """
#         Copied from the lap (very slow!)
#     """
#     hists = []
#     for frame in frames:
#         try:
#             chans = cv2.split(frame)
#         except:
#             print(frame)
#             x = 1/0
#             print(x)
#         color_hist = np.zeros((256,len(chans)))
#         for i in range(len(chans)):
#             color_hist[:,i] = np.histogram(chans[i], bins=np.arange(256+1))[0]/float((chans[i].shape[0]*chans[i].shape[1]))
#         hists.append(color_hist)
#     return hists