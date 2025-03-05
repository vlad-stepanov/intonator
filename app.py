import os
import tempfile
import pandas as pd
import streamlit as st

from audio_processor import AudioProcessor
from audio_extractor import AudioExtractor
from speech_emotion_classifier import SpeechEmotionClassifier
from vad_processor import VADProcessor
from visualizer import Visualizer

# Словарь классов
INVERSE_LABEL_MAP = {
    0: "neutral",
    1: "happy",
    2: "sad",
    3: "angry",
    4: "fearful",
    5: "disgusted",
    6: "surprised"
}

class MainApp:
    def __init__(self, model_path: str):
        self.audio_processor = AudioProcessor()
        self.video_processor = AudioExtractor()
        self.vad_processor = VADProcessor()
        self.classifier = SpeechEmotionClassifier(model_path, INVERSE_LABEL_MAP)
        self.visualizer = Visualizer()

    def plot_segments(self, segments: list):
        """
        Функция для отображения графика с сегментами.
        segments: список кортежей (start, end, emotion)
        """
        fig = self.visualizer.prepare_plot(segments)
        st.pyplot(fig)

    def run(self):
        st.title("Классификация интонации и эмоций")

        mode = st.radio("Выберите режим работы:", ("Запись аудио", "Загрузка видео"))

        if mode == "Запись аудио":
            input_mode = st.radio("Выберите способ ввода:", ("Записать свой голос", "Загрузить аудиофайл"))

            if input_mode == "Записать свой голос":
                if st.button("Начать запись"):
                    st.write("Запись... Говорите!")
                    audio, sr = self.audio_processor.record_audio()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                        self.audio_processor.save_wav(temp_file.name, audio, sr)
                        temp_filename = temp_file.name

                    st.audio(temp_filename, format="audio/wav")
                    result = self.classifier.classify_audio(temp_filename)
                    st.write(f"Результат: {result}")
                    os.unlink(temp_filename)

            elif input_mode == "Загрузить аудиофайл":
                uploaded_file = st.file_uploader("Загрузите аудиофайл (wav, mp3)", type=["wav", "mp3"])
                if uploaded_file is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_file:
                        audio_file.write(uploaded_file.read())
                        audio_filename = audio_file.name

                    st.audio(audio_filename, format="audio/wav")
                    result = self.classifier.classify_audio(audio_filename)
                    st.write(f"Результат: {result}")
                    os.unlink(audio_filename)

        elif mode == "Загрузка видео":
            uploaded_file = st.file_uploader("Загрузите видеофайл (mp4)", type="mp4")
            vad_mode = st.selectbox("Выберите режим VAD (0 – мягкий, 3 – строгий)", [0, 1, 2, 3], index=3)
            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as video_file:
                    video_file.write(uploaded_file.read())
                    video_filename = video_file.name

                st.video(video_filename)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_file:
                    audio_filename = audio_file.name
                self.video_processor.extract_audio(video_filename, audio_filename)

                signal, sr, segments = self.vad_processor.apply_vad(audio_filename, vad_mode=vad_mode)
                results = self.classifier.classify_segments(signal, sr, segments)

                results_df = pd.DataFrame(results, columns=["Начало (с)", "Конец (с)", "Эмоция"])
                st.write(results_df)

                if results:
                    self.plot_segments(results)

                os.unlink(video_filename)
                os.unlink(audio_filename)

if __name__ == "__main__":
    model_path = "../model.pth"
    app = MainApp(model_path)
    app.run()
