import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def train_unimodal(X, y, output_path):
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
    model.fit(X, y)
    joblib.dump(model, output_path)
    return model
