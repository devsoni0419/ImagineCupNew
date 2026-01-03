import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

def train_late_fusion(speech_p, hand_p, gait_p, y, output_path):
    X = np.column_stack([speech_p, hand_p, gait_p])
    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, output_path)
    return model
