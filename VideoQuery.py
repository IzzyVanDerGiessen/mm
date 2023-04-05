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

    audio_samplerate, audio_samples = wav.read(video_path.split('.mp4')[0] + ".wav")
    audio_path = video_path.split('.mp4')[0] + ".wav"
    query_features = []

    if '.avi' in video_path:
        audio_samplerate, audio_samples = librosa.load(video_path.split('.avi')[0] + ".wav")
    for feature in features:
        if feature in VISUAL_FEATURES:
            query_features.append(compute_feature(feature, video_path = video_path, video_data = frames))
        if feature in AUDIO_FEATURES:
            query_features.append(compute_feature(feature, audio_path = audio_path, audio_data = audio_samples.astype('float32'), samplerate = audio_samplerate))

    
    results = {}
    videos = os.listdir(Database.FULL_VIDEOS_PATH)
    videos = filter(lambda x: ('.mp4' in x), videos) #filters our '.wav' files in the matching videos

    for video in videos:
        print(video)
        
        out = []
        test_len = vid_len(Database.FULL_VIDEOS_PATH + video)
        test_fps = get_fps(Database.FULL_VIDEOS_PATH + video)

        if test_fps == 0:
            continue

        if '.avi' in video: 
            test_audio_samplerate, test_audio_samples = wav.read(Database.FULL_VIDEOS_PATH + video.split('.avi')[0] + ".wav")
        else: 
            test_audio_samplerate, test_audio_samples = wav.read(Database.FULL_VIDEOS_PATH + video.split('.mp4')[0] + ".wav")
        num_samples_per_frame = len(test_audio_samples) // test_len

        # step size in frames
        step_size = int((len(frames) / fps) * test_fps)

        cap = cv2.VideoCapture(Database.FULL_VIDEOS_PATH + video)
        for i in range(0, test_len-len(frames), step_size):
            video_data = None
            audio_data = None
                
            test_features = []

            for feature in features:                             
                if feature in  ["colorhists", "temporal_diff"]:
                    video_data = []
                    for j in range(i, i+len(frames)):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        video_data.append(frame)
                elif feature in ["audio_powers", "mfccs"]: 
                    audio_data = test_audio_samples[i:i+num_samples_per_frame].astype('float32')
                
                test_feature = compute_feature(feature, video_data = video_data, audio_data = audio_data, samplerate = test_audio_samplerate, num_frames = step_size)
                test_features.append(test_feature)
            score = feature_scorer(test_features, query_features)

            
            if score == score: #check if the number is not a nan 
                out.append((video + ": " + str(i/test_fps) + "-" + str((i+step_size)/test_fps), score))
            else:
                print(score)
            

        results[video] = sorted(out, key=lambda x: x[1])

    best = []
    for res in results.keys():
        best += results[res][:5]
    
    print(delimiter,"Query Item:(",  features,");", video_path, delimiter)
    print(sorted(best, key=lambda x: x[1])[:5], delimiter)
    print("Time taken:", time.time() - start, "seconds", delimiter)

def signature_pipeline(file_path, feature):
    return None 


def vid_len(video):
    cap = cv2.VideoCapture(video)
    vid_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return vid_len



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
                frames = getVideoFrames(audio_path)
                samplerate, samples = wav.read(audio_path.split('.mp4')[0] + ".wav")
                if '.avi' in audio_path:
                    samplerate, samples = wav.read(audio_path.split('.avi')[0] + ".wav")
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
