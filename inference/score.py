import json
import joblib
import numpy as np

def init():
    global model
    model = joblib.load("model/late_fusion.pkl")

def run(raw_data):
    data = json.loads(raw_data)

    speech = data["speech"]
    handwriting = data["handwriting"]
    gait = data["gait"]

    X = np.array(speech + handwriting + gait).reshape(1, -1)
    risk = model.predict_proba(X)[0][1]

    return {
        "risk_score": float(risk),
        "confidence": float(max(risk, 1 - risk)),
        "disclaimer": "This is NOT a medical diagnosis. For screening only."
    }
