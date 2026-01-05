import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump

from features.speechFeatures import load_audio_dataset
from features.handwritingFeatures import load_handwriting_dataset

Xa, ya = load_audio_dataset("data/audio")
Xh, yh = load_handwriting_dataset("data/drawing")

Xa0, Xa1 = Xa[ya == 0], Xa[ya == 1]
Xh0, Xh1 = Xh[yh == 0], Xh[yh == 1]

n0 = min(len(Xa0), len(Xh0))
n1 = min(len(Xa1), len(Xh1))

X0 = np.hstack([Xa0[:n0], Xh0[:n0]])
X1 = np.hstack([Xa1[:n1], Xh1[:n1]])

X = np.vstack([X0, X1])
y = np.array([0]*n0 + [1]*n1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    random_state=42
)

model.fit(X_train, y_train)

print("Classes:", model.classes_)
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

os.makedirs("models", exist_ok=True)
dump(model, "models/multimodal_pd_model.joblib")
