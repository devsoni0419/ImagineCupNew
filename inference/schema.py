input_schema = {
    "speech": [float, float, float, float],
    "handwriting": [float, float, float],
}

output_schema = {
    "risk_score": float,
    "confidence": float,
    "disclaimer": str
}
