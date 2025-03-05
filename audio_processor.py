import sounddevice as sd
import numpy as np
import wave

class AudioProcessor:
    def __init__(self):
        pass

    def record_audio(self, duration: int = 5, samplerate: int = 16000) -> tuple[np.ndarray, int]:
        audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
        sd.wait()
        return audio_data, samplerate

    def save_wav(self, filename: str, data: np.ndarray, samplerate: int):
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16 бит
            wf.setframerate(samplerate)
            wf.writeframes(data.tobytes())
