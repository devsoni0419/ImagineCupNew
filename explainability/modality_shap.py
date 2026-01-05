import numpy as np

AUDIO_FEATURE_COUNT = 17
HANDWRITING_FEATURE_COUNT = 4

def modality_contributions(shap_values):
    values = shap_values[0][:, 1]

    audio = np.sum(np.abs(values[:AUDIO_FEATURE_COUNT]))
    handwriting = np.sum(np.abs(values[AUDIO_FEATURE_COUNT:]))

    total = audio + handwriting

    return {
        "speech": audio / total,
        "handwriting": handwriting / total
    }
