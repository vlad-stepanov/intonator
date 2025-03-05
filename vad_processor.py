import webrtcvad
import librosa
import numpy as np

class VADProcessor:
    def __init__(self, vad_mode: int = 3):
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(vad_mode)  # режим VAD от 0 до 3

    def apply_vad(self, audio_path: str, vad_mode, sr: int = 16000) -> tuple[np.ndarray, int, list]:
        self.vad.set_mode(vad_mode)  # режим VAD от 0 до 3
        signal, _ = librosa.load(audio_path, sr=sr)
        frame_length = int(sr * 0.03)  # 30 мс
        speech_segments = []
        start_time = None

        for i in range(0, len(signal), frame_length):
            frame = signal[i:i + frame_length]
            if len(frame) < frame_length:
                break
            frame_int16 = np.int16(frame * 32767)
            is_speech = self.vad.is_speech(frame_int16.tobytes(), sr)
            if is_speech and start_time is None:
                start_time = i / sr
            elif not is_speech and start_time is not None:
                end_time = i / sr
                speech_segments.append((start_time, end_time))
                start_time = None

        if start_time is not None:
            speech_segments.append((start_time, len(signal) / sr))
        return signal, sr, speech_segments
