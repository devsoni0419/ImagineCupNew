import shap
import joblib
import numpy as np

def explain_model(model_path, X_sample):
    model = joblib.load(model_path)
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)

    return shap_values.values.tolist()
