audio_feature_names = (
    [f"MFCC_{i}" for i in range(13)]
    + ["Jitter", "Shimmer", "Pitch", "RMS"]
)

handwriting_feature_names = [
    "Contour_Length_Mean",
    "Contour_Length_STD",
    "Contour_Area_Mean",
    "Contour_Area_STD"
]

FEATURE_NAMES = audio_feature_names + handwriting_feature_names
