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

    audio_samplerate, audio_samples = wav.read(video_path.split('.mp4')[0] + ".wav")
    if 'BlackKnight' in video_path:
        audio_samplerate, audio_samples = librosa.load(video_path.split('.avi')[0] + ".wav")
    if feature in  ["colorhists", "temporal_diff"]:
        query_feature = compute_feature(feature, path = video_path, data = frames)
    if feature in  ["mfccs", "audio_powers"]:
        query_feature = compute_feature(feature, path = video_path, data = audio_samples.astype('float32'), samplerate = audio_samplerate)


    results = {}
    videos = os.listdir(Database.FULL_VIDEOS_PATH)

    results = force(videos, len(frames), fps, query_feature)

    print("Query Item", "(" + feature +"):", video_path)
    print(sorted(results, key=lambda x: x[1])[:5])
    print("Time taken:", time.time() - start, "seconds")

def signature_pipeline(video_path, feature, k=10):
    """
        Our version of the pipeline.
    """
    start = time.time()
    Database.loadCroppedVideos()

    frames = getVideoFrames(video_path)
    fps = get_fps(video_path)

    audio_samplerate, audio_samples = wav.read(video_path.split('.mp4')[0] + ".wav")
    if 'BlackKnight' in video_path:
        audio_samplerate, audio_samples = librosa.load(video_path.split('.avi')[0] + ".wav")
    if feature in  ["colorhists", "temporal_diff"]:
        query_feature = compute_feature(feature, path = video_path, data = frames)
    if feature in  ["mfccs", "audio_powers"]:
        query_feature = compute_feature(feature, path = video_path, data = audio_samples.astype('float32'), samplerate = audio_samplerate)


    videos = os.listdir(Database.FULL_VIDEOS_PATH)

    sign_results = {}
    for video in videos:
        if '.wav' in video:
            continue
        if 'BlackKnight' in video:
            continue
        #segments = filter(lambda x: x.startswith(video[:-4]), os.listdir(Database.CROPPED_VIDEOS_PATH))
        sign_results[video[:-4]] = []

        db = Database.cropped_signs[video[:-4]][feature]
        for test_feature in db:
            score = np.abs(test_feature - query_feature).sum()
            if score == score:
                sign_results[video[:-4]].append(score)

    sign_res = {k: min(v) for k, v in sign_results.items()}
    kvideos = list(map(lambda x: x+".mp4", sorted(sign_res.keys(), key = lambda x: sign_res[x])[:k]))
    results = force(kvideos, len(frames), fps, query_feature)

    print("Query Item", "(" + feature +"):", video_path)
    print(sorted(results, key=lambda x: x[1])[:5])
    print("Time taken:", time.time() - start, "seconds")

def force(videos, len_frames, fps, query_feature):
    results = []
    for video in videos:
        if '.wav' in video:
            continue
        print(video)

        test_len = vid_len(Database.FULL_VIDEOS_PATH + video)
        test_fps = get_fps(Database.FULL_VIDEOS_PATH + video)

        if test_fps == 0:
            continue


        if '.avi' in video:
            test_samplerate, test_samples = wav.read(Database.FULL_VIDEOS_PATH + video.split('.avi')[0] + ".wav")
        else:
            test_samplerate, test_samples = wav.read(Database.FULL_VIDEOS_PATH + video.split('.mp4')[0] + ".wav")
        num_samples_per_frame = len(test_samples) // test_len

        # step size in frames
        vid_size = (len_frames / fps) * test_fps
        step_size = int(vid_size / 2)

        frames = getVideoFrames(Database.FULL_VIDEOS_PATH + video)
        sample = []
        for i in range(0, test_len-len_frames, step_size):

            # to extend the implementation for both video and audio
            if feature in  ["colorhists", "temporal_diff"]:
                sample += frames[i+len(sample) : i+len_frames]

            elif feature in ["audio_powers", "mfccs"]:
                sample = test_samples[i:i+num_samples_per_frame].astype('float32')

            test_feature = compute_feature(feature, data = sample, samplerate = test_samplerate, num_frames = len_frames)
            lengthLimiter = min(len(test_feature), len(query_feature)) #since sometimes we end up with weird numbers of frames (cuz of end of vid?)
            score = np.abs(test_feature[:lengthLimiter] - query_feature[:lengthLimiter]).sum()
            if score == score:
                results.append((video + ": " + str(i/test_fps) + "-" + str((i+vid_size)/test_fps), score))
            else:
                pass
            sample = sample[min(step_size, len_frames):]
    return results

def compute_feature(feature, path = None, data = None, samplerate = None, num_frames = None):
    match feature:
        case "colorhists" :
            if path != None:
                frames = getVideoFrames(path)
            if type(data) != None:
                frames = data
            return sign_methods[feature](frames)
        case "mfccs":
            if type(data) != None:
                return sign_methods[feature](data, samplerate)
            return None
        case "audio_powers":
            if path != None:
                frames = getVideoFrames(path)
                samplerate, samples = wav.read(path.split('.mp4')[0] + ".wav")
                if 'BlackKnight' in path:
                    samplerate, samples = wav.read(path.split('.avi')[0] + ".wav")
                return sign_methods[feature](samplerate, samples, len(frames))
            if type(data) != None:
                return sign_methods[feature](samplerate, data, num_frames)

            return None # need to figure out how to do audio data

        case "temporal_diff":
            if path != None:
                frames = getVideoFrames(path)

            if type(data) != None:
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
