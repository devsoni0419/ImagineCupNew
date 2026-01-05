import cv2
import numpy as np
import os

def extract_handwriting_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))

    edges = cv2.Canny(img, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_lengths = [cv2.arcLength(cnt, True) for cnt in contours]
    contour_area = [cv2.contourArea(cnt) for cnt in contours]

    return np.array([
        np.mean(contour_lengths),
        np.std(contour_lengths),
        np.mean(contour_area),
        np.std(contour_area)
    ])


def load_handwriting_dataset(base_path):
    X, y = [], []

    for label, cls in enumerate(["Healthy", "Parkinsons"]):
        folder = os.path.join(base_path, cls)
        for file in os.listdir(folder):
            if file.endswith(".png"):
                X.append(extract_handwriting_features(os.path.join(folder, file)))
                y.append(label)

    return np.array(X), np.array(y)
