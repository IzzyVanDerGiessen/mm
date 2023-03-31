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


# the keys correspond to the folder names
def mfccs(audio, samplerate):
    # library implementation, idk what else to use tbh
    features = librosa.feature.mfcc(y=audio, sr=samplerate, n_mfcc=13)

    return features


def getVideoFrames(video_path):
    #cap = cv2.VideoCapture(video_path)
    #ret = True
    #frames = []
    #i = 0
    #while ret:
    #    ret, frame = cap.read()
    #    frames.append(frame)
    #    i += 1
    #    print(i)
    #cap.release()
    #print("End of vid!")
    #frames.pop()
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
    for frame in frames:
        if not last:
            continue
        out.append(temporal_diff_frames(frame, last))
        last = frame
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

def mm_mfcc_colorhist(frames, audio, samplerate):
    colorhists= colorhist(frames)
    mfcc = mfccs(audio.astype('float32'), samplerate)


    return [colorhists, mfcc]

sign_methods = {
    "colorhists": colorhist,
    "mfccs": mfccs,
    "temporal_diff": temporal_difference,
    "audio_powers": audio_signal_power
}
