import streamlit as st
import requests
import json

st.title("Early Parkinson's Risk Screening")
st.warning("Not a medical diagnosis.")

speech = st.text_input("Speech features (comma separated)")
hand = st.text_input("Handwriting features")


if st.button("Analyze"):
    payload = {
        "speech": list(map(float, speech.split(","))),
        "handwriting": list(map(float, hand.split(","))),
    }

    response = requests.post(
        "<AZURE_ENDPOINT_URL>",
        headers={"Authorization": "Bearer <API_KEY>"},
        json=payload
    )

    st.json(response.json())
