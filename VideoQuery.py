import Database
import os
from Signatures import *
import numpy as np
import cv2
import argparse
import time
from Scorer import *

VISUAL_FEATURES = ["colorhists", "temporal_diff"]
AUDIO_FEATURES = ["audio_powers", "mfccs"]
delimiter = "\n==============================================================\n"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type = str, default = "./videos_cropped/British_Plugs_Are_Better_from_0.0_to_5.0).mp4")
    parser.add_argument("--pipeline", type = str, default = "brute_force")
    parser.add_argument('--features','--list', nargs='+', help='<Required> Set flag', required=True)
    return parser.parse_args()

def brute_force_pipeline(video_path, features):
    """
        The brute force version of the query.
    """
    start = time.time()

    frames = getVideoFrames(video_path)
    fps = get_fps(video_path)

    
    audio_path = video_path[:-4] + ".wav"
    query_features = []
    audio_samples, audio_samplerate = librosa.load(video_path[:-4] + ".wav")

    for feature in features:
        if feature in VISUAL_FEATURES:
            query_features.append(compute_feature(feature, video_path = video_path, video_data = frames))
        if feature in AUDIO_FEATURES:
            query_features.append(compute_feature(feature, video_path = video_path, audio_path = audio_path, audio_data = audio_samples.astype('float32'), samplerate = audio_samplerate))

    results = {}
    videos = os.listdir(Database.FULL_VIDEOS_PATH)
    videos = filter(lambda x: ('.mp4' in x), videos) #filters our '.wav' files in the matching videos

    results = force(videos, len(frames), fps, query_features)

    print(delimiter, "Query Item", features,":" , video_path, delimiter)
    for key, value in sorted(results, key=lambda x: x[1])[:5]:
        print(key, value)
    print(delimiter, "Time taken:", time.time() - start, "seconds")

def signature_pipeline(video_path, features, k=5):
    """
        Our version of the pipeline.
    """
    start = time.time()
    Database.loadDatabase()
    
    audio_samples, audio_samplerate = librosa.load(video_path[:-4] + ".wav")
    audio_path = video_path.split('.mp4')[0] + ".wav"
    query_features = []

    query_features = []
    for feature in features:
        if feature in  ["colorhists", "temporal_diff"]:
            query_feature = compute_feature(feature, video_path = video_path, video_data = frames)
            query_features.append(query_feature)
        if feature in  ["mfccs", "audio_powers"]:
            query_feature = compute_feature(feature, video_path = video_path, audio_path = audio_path, audio_data = audio_samples.astype('float32'), samplerate = audio_samplerate)
            query_features.append(query_feature)

    videos = os.listdir(Database.FULL_VIDEOS_PATH)
    videos = filter(lambda x: ('.mp4' in x), videos)
    sign_results = {}
    for video in videos:
        sign_results[video[:-4]] = []
        

        test_features = []
        for j, feature in enumerate(features):
            
            db = Database.cropped_signs[video[:-4]][feature] + [Database.full_signs[video[:-4]][feature]]
            for i, feat in enumerate(db):
                if j == 0:
                    test_features.append([])
                test_features[i].append(feat)
                
            

        for tests in test_features:
            
            score = feature_scorer(tests, query_features)
            if score == score:
                sign_results[video[:-4]].append(score)

    sign_res = {k: min(v) for k, v in sign_results.items()}
    print(delimiter, list(map(lambda x: x+".mp4", sorted(sign_res.keys(), key = lambda x: sign_res[x]))), delimiter)

    #n_list = list(map(lambda x: x+".mp4", sorted(sign_res.keys(), key = lambda x: sign_res[x])))
    kvideos = list(map(lambda x: x+".mp4", sorted(sign_res.keys(), key = lambda x: sign_res[x])[:k]))
    results = force(kvideos, len(frames), get_fps(video_path), query_features)

    print(delimiter, "Query Item", features,":" , video_path, delimiter)
    for key, value in sorted(results, key=lambda x: x[1])[:5]:
        print(key, value)
    print(delimiter, "Time taken:", time.time() - start, "seconds") 

def force(videos, len_frames, fps, query_features):
    results = []
    videos = filter(lambda x: ('.mp4' in x), videos)
    for video in videos:
        print(video)
        intermediary_results = []

        test_len = vid_len(Database.FULL_VIDEOS_PATH + video)
        test_fps = get_fps(Database.FULL_VIDEOS_PATH + video)

        if test_fps == 0:
            continue

        
        test_samples, test_samplerate = librosa.load(Database.FULL_VIDEOS_PATH + video[:-4] + ".wav")
        num_samples_per_frame = len(test_samples) // test_len

        # step size in frames
        vid_size = (len_frames / fps) * test_fps
        step_size = int(vid_size / 2)
        frames = getVideoFrames(Database.FULL_VIDEOS_PATH + video)

        sample = []
        for i in range(0, test_len-len_frames, step_size):
            test_features = []
            for feature in features:
                # to extend the implementation for both video and audio
                if feature in  ["colorhists", "temporal_diff"]:
                    sample = frames[i: i+len_frames]
                elif feature in ["audio_powers", "mfccs"]:
                    sample = test_samples[i * num_samples_per_frame: (i + len_frames) * num_samples_per_frame]

                test_feature = compute_feature(feature, audio_data = sample, video_data = sample, samplerate = test_samplerate, num_frames = len_frames)
                test_features.append(test_feature)
                
            score = feature_scorer(test_features, query_features)
            
            if score == score:
                intermediary_results.append((video + ": " + str(i/test_fps) + "-" + str((i+vid_size)/test_fps), score))
            else:
                pass
            sample = sample[min(step_size, len_frames):]
        results.append(sorted(intermediary_results, key=lambda x: x[1])[0])
    return results

def compute_feature(feature, video_path = None, audio_path = None, video_data = None, audio_data = None, samplerate = None, num_frames = None):
    match feature:
        case "colorhists" :
            if video_path != None:
                frames = getVideoFrames(video_path)
            if type(video_data) != None:
                frames = video_data
            return sign_methods[feature](frames)
        case "mfccs":
            if type(audio_data) == None:
                raise Exception("Error when matchig MFCCs!")
            return sign_methods[feature](audio_data, samplerate)
        case "audio_powers":
            if audio_path != None:
                frames = getVideoFrames(video_path)
                samplerate, samples = wav.read(video_path.split('.mp4')[0] + ".wav")
                if '.avi' in audio_path:
                    samplerate, samples = wav.read(video_path.split('.avi')[0] + ".wav")
                return sign_methods[feature](samplerate, samples, len(frames))
            if type(audio_data) != None:
                return sign_methods[feature](samplerate, audio_data, num_frames)

            raise Exception("Error when matching audio powers!")

        case "temporal_diff":
            if video_path != None:
                frames = getVideoFrames(video_path)
                return sign_methods[feature](frames)

            if video_data == None:
                raise Exception("Error when matching temporal difference!")

            frames = video_data
            return sign_methods[feature](frames)
        case _ :
             raise Exception("==================\n An error in the matching occured :) \n ==================")



if __name__ == '__main__':
    args = get_args()
    file_path = args.filepath
    pipeline = args.pipeline
    features = args.features
    frames = getVideoFrames(file_path)
    samplerate, audio = wav.read(file_path.split('.mp4')[0] + '.wav')
    if pipeline == "brute_force":
        brute_force_pipeline(file_path, features)
    elif pipeline == "signatures":
        signature_pipeline(file_path, features)
