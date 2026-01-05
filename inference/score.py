import os
import numpy as np
from joblib import load
from features.speechFeatures import extract_audio_features
from features.handwritingFeatures import extract_handwriting_features

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "multimodal_pd_model.joblib")

model = load(MODEL_PATH)

def predict_with_features(audio_path, image_path):
    audio_feat = extract_audio_features(audio_path)
    img_feat = extract_handwriting_features(image_path)

    X = np.hstack([audio_feat, img_feat]).reshape(1, -1)

    prob = model.predict_proba(X)[0][1]
    return prob, X
