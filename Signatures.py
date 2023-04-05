import cv2
import numpy as np
import scipy.io.wavfile as wav
import librosa
import sys
import pims

database_path = "database/signatures/"
sign_types = ["colorhists", "mfccs", "temporal_diff", "audio_powers"]
#sign_types = ["colorhists"]

def colorhist(frames):
    avg_hists = np.zeros((3, 256))
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        c1 = frame[:,:,0]
        c2 = frame[:,:,1]
        c3 = frame[:,:,2]
        cs = [c1, c2, c3]
        for i, c in enumerate(cs):
            hist = np.bincount(c.flatten(), None, 256)
            avg_hists[i] += hist
    return (avg_hists / len(frames)).flatten()


def mfccs(audio, samplerate):
    # library implementation, idk what else to use tbh
    features = librosa.feature.mfcc(y=audio, sr=samplerate, n_mfcc=13, n_fft=512)
    return features


def getVideoFrames(video_path):
    return pims.Video(video_path)

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


def temporal_difference(frames):
    out = []
    last = None
    for i in range(0, len(frames) - 1):
        frame = frames[i]
        next = frames[i+1]
        out.append(temporal_diff_frames(frame, next))
    return np.array(out)


def audio_signal_power(samplerate, samples, num_frames):
    time = samples.shape[0] / samplerate
    frame_rate = num_frames // time

    T = int(samplerate // frame_rate)

    out = []
    i = 0

    for j in range(num_frames):
        out.append(np.sum(np.square(samples[i:i+T]))/T)
        i += int(T)

    return np.array(out)



sign_methods = {
    "colorhists": colorhist,
    "temporal_diff": temporal_difference,
    "mfccs": mfccs,
    "audio_powers": audio_signal_power
}
