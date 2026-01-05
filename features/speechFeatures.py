import librosa
import numpy as np
import os

def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    jitter = np.std(librosa.feature.zero_crossing_rate(y))
    shimmer = np.std(y)
    pitch = np.mean(librosa.yin(y, fmin=50, fmax=300))
    rms = np.mean(librosa.feature.rms(y=y))

    return np.hstack([mfcc, jitter, shimmer, pitch, rms])


def load_audio_dataset(base_path):
    X, y = [], []

    for label, cls in enumerate(["healthy", "parkinsons"]):
        folder = os.path.join(base_path, cls)
        for file in os.listdir(folder):
            if file.endswith((".wav", ".flac")):
                features = extract_audio_features(os.path.join(folder, file))
                X.append(features)
                y.append(label)

    return np.array(X), np.array(y)
