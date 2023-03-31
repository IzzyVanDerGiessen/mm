import cv2
import numpy as np
import scipy.io.wavfile as wav
import librosa

database_path = "database/signatures/"
sign_types = ["colorhists", "mfccs", "temporal_diff", "audio_powers"]

# def signColorhists(video_path):
#     print(video_path)
#     cap = cv2.VideoCapture(video_path)
#     ret = True
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     i = 0
#     avg_hists = np.zeros(256)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         hist = np.bincount(gray.flatten(), None, 256)
#         avg_hists += hist
#         i += 1
#     avg_hists /= i
#     return avg_hists

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


def get_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    images = []
    while (cap.isOpened()):
        retVal, frame = cap.read()
        if retVal == False:
            break
        images.append(frame)

    return images

def getVideoFrames(video_path):
    cap = cv2.VideoCapture(video_path)
    ret = True
    frames = []
    while ret:
        ret, frame = cap.read()
        frames.append(frame)
    cap.release()

    frames.pop()
    return frames


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
    for i in range(len(frames) - 1):
        out.append(temporal_diff_frames(frames[i+1], frames[i]))
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
    "mfccs": mfccs,
    "temporal_diff": temporal_difference,
    "audio_powers": audio_signal_power
}



