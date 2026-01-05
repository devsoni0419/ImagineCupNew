import shap
import os
from joblib import load

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "multimodal_pd_model.joblib")

model = load(MODEL_PATH)
explainer = shap.TreeExplainer(model)

def explain_prediction(X):
    shap_values = explainer.shap_values(X)
    return shap_values
