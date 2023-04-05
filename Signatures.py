import cv2
import numpy as np
import scipy.io.wavfile as wav
import librosa
import sys
import pims

database_path = "database/signatures/"
sign_types = ["colorhists", "mfccs", "temporal_diff", "audio_powers"]


def colorhist(frames):

    avg_hists = np.zeros(256)
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = np.bincount(gray.flatten(), None, 256)
        avg_hists += hist

    return avg_hists / len(frames)


def mfccs(audio_path='./videos/BlackKnight.wav'):
    audio, sample_rate = librosa.load(audio_path)
    # library implementation, idk what else to use tbh
    features = librosa.feature.mfcc(y=audio, sr=sample_rate)
    return features

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


def temporal_difference(video, path = True):
    out = []
    if path:
        video = get_video_frames(video)
    for i in range(len(video) - 1):
        out.append(temporal_diff_frames(video[i+1], video[i]))
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
    "colorhists": colorhist,
    "temporal_diff": temporal_difference,
    "mfccs": mfccs,
    "audio_powers": audio_signal_power
}
