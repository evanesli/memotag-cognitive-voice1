import streamlit as st
import whisper
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import librosa
import re
from sklearn.ensemble import IsolationForest

# Load Whisper model (small for speed — replace with 'base' or 'medium' if needed)
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("small")

model = load_whisper_model()

# Feature Extraction Functions
def extract_features(audio_path, transcript):
    y, sr = librosa.load(audio_path, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)

    # 1. Speech Rate
    word_count = len(transcript.split())
    speech_rate = word_count / duration

    # 2. Pauses
    intervals = librosa.effects.split(y, top_db=30)
    pause_durations = []
    for i in range(1, len(intervals)):
        pause = (intervals[i][0] - intervals[i-1][1]) / sr
        pause_durations.append(pause)
    pause_count = sum(1 for p in pause_durations if p > 0.3)

    # 3. Hesitation Words
    hesitations = re.findall(r"\b(uh|um|like|you know)\b", transcript.lower())
    hesitation_count = len(hesitations)

    # Final Feature Vector
    return np.array([speech_rate, pause_count, hesitation_count]), {
        "speech_rate": speech_rate,
        "pause_count": pause_count,
        "hesitation_count": hesitation_count
    }

# Risk Scoring using Isolation Forest
def score_risk(X):
    clf = IsolationForest(contamination=0.2, random_state=42)
    clf.fit(np.vstack([X, X + np.random.normal(0, 0.5, size=X.shape)]))  # simulate normal data
    return clf.predict([X])[0]  # -1 = at risk, 1 = normal

# Streamlit App
st.set_page_config(page_title="MemoTag Cognitive Voice Analyzer")
st.title("🧠 MemoTag - Cognitive Risk from Voice")

uploaded_file = st.file_uploader("Upload a short voice recording (.wav)", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("Transcribing with Whisper..."):
        result = model.transcribe(tmp_path)
        transcript = result["text"]
        st.subheader("Transcript")
        st.write(transcript)

    with st.spinner("Extracting features..."):
        features, breakdown = extract_features(tmp_path, transcript)

    st.subheader("Feature Breakdown")
    st.write(breakdown)

    risk_label = score_risk(features)
    st.subheader("🩺 Cognitive Risk Score")
    if risk_label == -1:
        st.error("High Risk: Abnormal speech patterns detected")
    else:
        st.success("Low Risk: No strong indicators of cognitive decline")

    # Visualizations
    st.subheader("Feature Trends")
    fig, ax = plt.subplots()
    ax.bar(breakdown.keys(), breakdown.values(), color=["orange", "skyblue", "red"])
    ax.set_ylabel("Value")
    ax.set_title("Speech Feature Analysis")
    st.pyplot(fig)

    os.remove(tmp_path)
