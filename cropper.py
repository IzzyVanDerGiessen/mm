import cv2 
import os 
import argparse
import scipy.io.wavfile as wav
import moviepy.editor as mp


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type = bool, default = False)
    parser.add_argument("--video", type = bool, default = False)
    parser.add_argument("--input", type = bool, default = False)
    parser.add_argument("--input_path", type = str, default = "./input-videos/Why_Does_Nighttime_Smartphone_Footage_Look_All_Flickery_from_15.0_to_26.0.mp4")
    parser.add_argument("--cliplength", type = int, default = 5)
    return parser.parse_args()

def crop_videos(clip_duration):
    video_names = []
    
    for name in os.listdir('./videos'):
        if '.mp4' in name and not (name.split('.')[0] in os.listdir('./videos_cropped')): 
            video_names.append(name)


    
    for k, video in enumerate(video_names):
        capture = cv2.VideoCapture('./videos/' + video)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        duration_of_video = total_frames // fps #duration of video in seconds 
        num_clips = int(duration_of_video // clip_duration)



        for i in range(num_clips):
            video_beginning = i*clip_duration
            clip_name = (video.split('.')[0] + '_from_{start:.1f}_to_{end:.1f}).mp4').format(start = video_beginning, end = video_beginning + clip_duration)
            start_frame = int(i * clip_duration * fps)
            end_frame = start_frame + int(clip_duration * fps)
            clip_frames = []

            for j in range(start_frame, end_frame):
                ret, frame = capture.read()
                if ret: 
                    clip_frames.append(frame)
            out = cv2.VideoWriter('./videos_cropped/' + clip_name, cv2.VideoWriter_fourcc(*"mp4v"),capture.get(cv2.CAP_PROP_FPS),
            (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))

            for f in clip_frames:
                out.write(f)
            out.release()
            print(clip_name)
        
        capture.release()

def crop_audio(clip_duration):
    audio_names = []
    for name in os.listdir('./videos'):
        if '.wav' in name: 
            print(name)
            audio_names.append(name)
            print(name)

    
    for k, audio in enumerate(audio_names):
        samplerate, samples = wav.read("./videos/" + audio)
        step_size = samplerate * clip_duration


      
        for j in range(0, len(samples), 1):
            data = samples[j:j + step_size]
            audio_beginning = j // samplerate
            clip_name = "./videos_cropped/" + (audio.split('.')[0] + '_from_{start:.1f}_to_{end:.1f}).wav').format(start = audio_beginning, end = audio_beginning + clip_duration)
            wav.write(clip_name, samplerate, data)

def process_input(filepath):
    video = mp.VideoFileClip(filepath)
    audio_array = video.audio.to_soundarray().astype('float32')
    wav.write(filepath.split('.mp4')[0]+".wav", video.audio.fps, audio_array)




if __name__ == '__main__':
    output_path = './videos_cropped'
    args = get_args()

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    clip_duration = args.cliplength #duration of the clip in seconds (5 by default)

    if args.video:
        crop_videos(clip_duration)
    if args.audio:
        crop_audio(clip_duration)
    if args.input:
        process_input(args.input_path)


    