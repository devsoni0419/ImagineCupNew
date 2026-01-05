def clinical_explanation(risk, modality):
    statements = []

    if modality["speech"] > 0.6:
        statements.append(
            "speech biomarkers such as pitch variability and vocal stability showed notable deviations"
        )

    if modality["handwriting"] > 0.4:
        statements.append(
            "handwriting patterns exhibited irregular motor control characteristics"
        )

    if not statements:
        statements.append(
            "no strong abnormal motor or speech patterns were detected"
        )

    return (
        "The model’s risk estimate is influenced by "
        + " and ".join(statements)
        + ". This result represents an early screening signal only and is not a medical diagnosis."
    )
