import streamlit as st
import os
import json

st.title("ADAS ML Platform Mini Dashboard")

metrics_path = "artifacts/metrics/metrics.json"

if os.path.exists(metrics_path):
    with open(metrics_path, "r") as file:
        metrics = json.load(file)

    st.subheader("Model Metrics")
    st.write(f"Accuracy: {metrics['accuracy']:.4f}")
    st.write(f"Precision: {metrics['precision']:.4f}")
    st.write(f"Recall: {metrics['recall']:.4f}")
    st.write(f"F1 Score: {metrics['f1_score']:.4f}")

    st.subheader("Confusion Matrix")
    st.write(metrics["confusion_matrix"])
else:
    st.warning("Metrics file not found. Please run evaluation first.")