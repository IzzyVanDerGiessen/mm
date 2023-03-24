import cv2 
import os 



if __name__ == '__main__':
    video_names = []
    output_path = './videos_cropped'

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    clip_duration = 5 #duration of the clip in seconds 
    print(os.listdir('./videos_cropped'))

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
        
                    




