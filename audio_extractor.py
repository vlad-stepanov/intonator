from moviepy import *

class AudioExtractor:
    def __init__(self):
        pass

    def extract_audio(self, video_path: str, audio_path: str, fps: int = 16000):
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, fps=fps)
