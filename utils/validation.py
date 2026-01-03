def validate_features(features, expected_len):
    if features is None or len(features) != expected_len:
        raise ValueError("Invalid feature vector length")
