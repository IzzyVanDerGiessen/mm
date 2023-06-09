import os
import numpy as np
import shutil
from Signatures import *
import librosa


PATH = "./database/signatures/"
FULL_VIDEOS_PATH = "./videos/"
CROPPED_VIDEOS_PATH = "./videos_cropped/"

def createDirectories(videos_folder, cropped_videos=False):
    videos = os.listdir(videos_folder)
    for video in videos:
        if '.wav' in video:
            continue
        print(video)


        if cropped_videos:
            spl = video.split("_from_")
            video_name = spl[0]
            segment_name = spl[1][:-5]
        else:
            video_name = video[:-4]
            segment_name = "full"

        video_path = videos_folder + video
        frames = getVideoFrames(video_path)

        for sign_type in sign_types:
            if not os.path.exists(PATH + sign_type):
                os.makedirs(PATH + sign_type)
            signature = None

            data_path = PATH + sign_type + "/" + video_name
            if not os.path.exists(data_path):
                os.makedirs(data_path)

            match sign_type:
                case "colorhists" :
                    signature = sign_methods[sign_type](frames)

                case "mfccs":
                    audio, sample_rate = librosa.load(video_path[:-4] + ".wav")
                    signature = sign_methods[sign_type](audio, sample_rate)

                case "audio_powers":
                    audio, sample_rate = librosa.load(video_path[:-4] + ".wav")
                    signature = sign_methods[sign_type](sample_rate, audio, len(frames))

                case "temporal_diff":
                    signature = sign_methods[sign_type](frames)


                case _ :
                    print("An error in the matching occured :)")
                    return -1

            # f = open(data_path + "/" + segment_name + ".txt", "wb")
            # f.write(signature.tobytes())
            # f.close()
            np.savetxt(data_path + "/" + segment_name + ".txt", signature)


full_signs = {}
cropped_signs = {}

def loadFullVideos():
    videos = os.listdir(FULL_VIDEOS_PATH)
    for video in videos:
        full_signs[video[:-4]] = {}
        for sign_method in sign_methods.keys():
            sign_file = PATH + sign_method + "/" + video[:-4] + "/full.txt"

            try:
                # with open(sign_file, "rb") as f:
                #     compare_sign_bts = f.read()
                #     compare_sign = np.frombuffer(compare_sign_bts, dtype=np.float32)
                #     full_signs[video[:-4]][sign_method] = compare_sign
                full_signs[video[:-4]][sign_method] = np.loadtxt(sign_file)
                
            except:
                print(sign_file)

def loadCroppedVideos():
    videos = os.listdir(CROPPED_VIDEOS_PATH)
    for video in videos:
        spl = video.split("_from_")
        video_name = spl[0]
        if video_name == 'BlackKnight':
            continue
        segment_name = spl[1][:-5]
        if video_name not in cropped_signs:
            cropped_signs[video_name] = {}
        for sign_method in sign_methods.keys():
            if sign_method not in cropped_signs[video_name]:
                cropped_signs[video_name][sign_method] = []
            sign_file = PATH + sign_method + "/" + video_name + "/" + segment_name + ".txt"
            # Some cropped videos have only a .wav file and not .mp4
            try:
                # with open(sign_file, "rb") as f:
                #     compare_sign_bts = f.read()
                #     compare_sign = np.frombuffer(compare_sign_bts)

                #     cropped_signs[video_name][sign_method].append(compare_sign)
                compare_sign = np.loadtxt(sign_file)
                cropped_signs[video_name][sign_method].append(compare_sign)
            except:
                continue

def loadDatabase():
    print("Start loading database...")
    loadFullVideos()
    loadCroppedVideos()
    print("Done!")

if __name__ == '__main__':
    refresh_database = True
    if refresh_database:
        # clean the previous database
        #shutil.rmtree("database/signatures")
        #os.makedirs("database/signatures")

        createDirectories(FULL_VIDEOS_PATH)
        createDirectories(CROPPED_VIDEOS_PATH, True)
    loadDatabase()
