import numpy as np

def extract_handwriting_features(x, y, pressure=None):
    velocity = np.mean(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    tremor = np.std(np.diff(x)) + np.std(np.diff(y))

    if pressure is None:
        pressure = np.random.normal(0.5, 0.1, len(x))

    return [velocity, tremor, np.mean(pressure)]
