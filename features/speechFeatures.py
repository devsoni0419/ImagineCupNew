import librosa
import numpy as np

def extract_speech_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)

    pitch = np.mean(librosa.yin(y, 75, 300))
    jitter = np.std(np.diff(y))
    shimmer = np.std(np.abs(y))
    pauses = np.sum(librosa.effects.split(y, top_db=30)) / len(y)

    return [pitch, jitter, shimmer, pauses]
