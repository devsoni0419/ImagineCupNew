def risk_band(prob):
    if prob < 0.35:
        return "Low Risk", "green"
    elif prob < 0.65:
        return "Moderate Risk", "orange"
    else:
        return "High Risk", "red"
