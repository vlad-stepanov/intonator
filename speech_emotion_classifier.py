import torch
import numpy as np
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import streamlit as st

# Словарь классов
INVERSE_LABEL_MAP = {
    0: "angry",
    1: "happy",
    2: "sad",
    3: "neutral",
    4: "fearful",
    5: "disgusted",
    6: "surprised"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SpeechEmotionClassifier:
    def __init__(self, model_path: str, label_map: dict):
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=len(label_map))
        self.state_dict = torch.load(model_path,map_location=device)
        self.model.load_state_dict(self.state_dict)
        self.model.to(device)
        self.model.eval()
        self.label_map = label_map

    def classify_audio(self, audio_path: str) -> str:
        
        # Загружаем аудио
        speech, sr = librosa.load(audio_path, sr=16000)
        
        # Приводим длину к 32000 сэмплов
        max_length = 32000
        if len(speech) > max_length:
            speech = speech[:max_length]
        else:
            speech = np.pad(speech, (0, max_length - len(speech)), 'constant')

        inputs = self.processor(speech, sampling_rate=16000, return_tensors='pt', padding=True, truncate=True, max_length=max_length)

        # Пропускаем через модель
        with torch.no_grad():
            outputs = self.model(inputs.input_values.to(device))
        logits = outputs.logits

        # Получаем предсказанный класс
        predicted_class = logits.argmax(dim=-1).item()
        return self.label_map[predicted_class]
    
    def classify_video(self, audio_data: np.ndarray, sr: int) -> str:
        # Обработка аудио через процессор
        inputs = self.processor(audio_data, sampling_rate=sr, return_tensors='pt', padding=True, truncation=True, max_length=sr)
        with torch.no_grad():
            logits = self.model(inputs.input_values.to(device)).logits
        predicted_class = logits.argmax(dim=-1).item()
        return self.label_map[predicted_class]

    def classify_segments(self, signal: np.ndarray, sr: int, segments: list) -> list:
        results = []
        for start, end in segments:
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segment = signal[start_sample:end_sample]
            result = self.classify_video(segment, sr)
            results.append((start, end, result))
        return results
