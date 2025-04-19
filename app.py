import streamlit as st
import whisper
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import librosa
import re
from sklearn.ensemble import IsolationForest

# ✅ This must be the first Streamlit command
st.set_page_config(page_title="MemoTag Cognitive Voice Analyzer", layout="centered")

# --- Load Whisper model ---
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("small")

model = load_whisper_model()

# --- Feature Extraction ---
def extract_features(audio_path, transcript):
    y, sr = librosa.load(audio_path, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)

    # Speech Rate
    word_count = len(transcript.split())
    speech_rate = word_count / duration if duration > 0 else 0

    # Pauses
    intervals = librosa.effects.split(y, top_db=30)
    pause_durations = []
    for i in range(1, len(intervals)):
        pause = (intervals[i][0] - intervals[i-1][1]) / sr
        pause_durations.append(pause)
    pause_count = sum(1 for p in pause_durations if p > 0.3)

    # Hesitation Markers
    hesitations = re.findall(r"\b(uh|um|like|you know)\b", transcript.lower())
    hesitation_count = len(hesitations)

    features = np.array([speech_rate, pause_count, hesitation_count])
    return features, {
        "Speech Rate (wpm)": round(speech_rate, 2),
        "Pause Count (>0.3s)": pause_count,
        "Hesitations (uh, um)": hesitation_count
    }

# --- Risk Scoring ---
def score_risk(X):
    clf = IsolationForest(contamination=0.2, random_state=42)
    # Simulate training on "normal" data
    fake_data = np.vstack([X, X + np.random.normal(0, 0.5, size=X.shape)])
    clf.fit(fake_data)
    return clf.predict([X])[0]  # -1 = risky

# --- Streamlit UI ---
st.title("🧠 MemoTag: Cognitive Risk from Voice")

uploaded_file = st.file_uploader("Upload a short voice recording (.wav)", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file)

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("Transcribing audio..."):
        result = model.transcribe(tmp_path)
        transcript = result["text"]
        st.subheader("📝 Transcript")
        st.write(transcript)

    with st.spinner("Analyzing speech features..."):
        features, breakdown = extract_features(tmp_path, transcript)

    st.subheader("📊 Feature Breakdown")
    st.json(breakdown)

    risk = score_risk(features)
    st.subheader("🧪 Cognitive Risk Score")

    if risk == -1:
        st.error("⚠️ High Risk: Abnormal speech patterns detected")
    else:
        st.success("✅ Low Risk: No major signs of cognitive decline")

    # Visualization
    st.subheader("📈 Feature Visualization")
    fig, ax = plt.subplots()
    ax.bar(breakdown.keys(), breakdown.values(), color=["orange", "skyblue", "crimson"])
    ax.set_ylabel("Value")
    ax.set_title("Speech Pattern Metrics")
    st.pyplot(fig)

    # Clean up temp file
    os.remove(tmp_path)
