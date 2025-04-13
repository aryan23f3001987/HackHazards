from transformers import pipeline
import librosa
import numpy as np
import tempfile

text_classifier = pipeline("sentiment-analysis")

# Text-based sentiment

def analyze_text_sentiment(text: str):
    result = text_classifier(text)[0]
    return result["label"], result["score"]

# Voice-based sentiment using basic audio features

def analyze_voice_features(wav_bytes: bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        tmp.flush()
        y, sr = librosa.load(tmp.name, sr=None)
        pitch = librosa.yin(y, fmin=50, fmax=300)
        avg_pitch = np.mean(pitch)
        energy = np.mean(np.square(y))

        # Very basic threshold logic
        if avg_pitch < 120 and energy < 0.01:
            return "negative", 0.85
        elif avg_pitch > 180:
            return "positive", 0.80
        else:
            return "neutral", 0.75

# Combine both sources

def merge_sentiments(text_sentiment: tuple, voice_sentiment: tuple) -> str:
    if "NEGATIVE" in [text_sentiment[0].upper(), voice_sentiment[0].upper()]:
        return "negative"
    elif "POSITIVE" in [text_sentiment[0].upper(), voice_sentiment[0].upper()]:
        return "positive"
    return "neutral"