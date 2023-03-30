import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import librosa
import shutil

database_path = "database/signatures/"
sign_types = ["colorhists", "mfccs", "temporal_diff", "audio_powers"]


def signColorhists(video_path):
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


def mfccs(audio_path='./videos/BlackKnight.wav'):
    audio, sample_rate = librosa.load(audio_path)
    # library implementation, idk what else to use tbh
    features = librosa.feature.mfcc(y=audio, sr=sample_rate)
    return features


def get_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    images = []
    while (cap.isOpened()):
        retVal, frame = cap.read()
        if retVal == False:
            break
        images.append(frame)

    return images


def temporal_diff_frames(frame, prev_frame):
    diff = 0
    for i in range(frame.shape[2]):
        diff = diff + \
            np.sum(np.abs(prev_frame[:, :, i].astype(
                'int16') - frame[:, :, i].astype('int16')))
    return diff


def temporal_difference(video_path):
    out = []
    frames = get_video_frames(video_path)
    for i in range(len(frames) - 1):
        out.append(temporal_diff_frames(frames[i+1], frames[i]))
    return np.array(out)


def audio_signal_power(audiopath):
    videopath = audiopath.split(".wav")[0] + ".mp4"
    if 'BlackKnight' in videopath:
        videopath = './videos/BlackKnight.avi'
    samplerate, samples = wav.read(audiopath)
    time = samples.shape[0] / samplerate

    frames = get_video_frames(videopath)
    num_frames = len(frames) + 1
    frame_rate = num_frames // time

    T = int(samplerate // frame_rate)

    out = []
    i = 0

    for frame in frames:
        out.append(np.sum(np.square(samples[i:i+T]))/T)
        i += int(T)

    return np.array(out)


sign_methods = {
    "colorhists": signColorhists,
    "mfccs": mfccs,
    "temporal_diff": temporal_difference,
    "audio_powers": audio_signal_power
}


# the keys correspond to the folder names
def createDirectories(videos_folder, cropped_videos=False):
    # clean the previous database
    # shutil.rmtree("database/signatures")
    # os.makedirs("database/signatures")

    videos = os.listdir(videos_folder)
    for video in videos:
        print(video)
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

            if (sign_type == 'mfccs' or sign_type == 'audio_powers') and ('.mp4' in video or '.avi' in video):
                continue
            if (sign_type == 'colorhists' or sign_type == 'temporal_diff') and ('.wav' in video):
                continue

            video_path = videos_folder + video
            signature = sign_methods[sign_type](video_path)

            f = open(data_path + "/" + segment_name + ".txt", "w")
            f.write(np.array_str(signature))
            f.close()


createDirectories("./videos/")
createDirectories("./videos_cropped/", True)
