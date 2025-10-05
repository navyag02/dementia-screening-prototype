import streamlit as st
import librosa
import numpy as np
import pandas as pd
import joblib
import tempfile
import random
import os
import string
from typing import Optional, Tuple

try:
    import assemblyai as aai  # type: ignore
except Exception:
    aai = None  # type: ignore

# Configure page early (must be before other Streamlit UI calls)
st.set_page_config(page_title="🧠 Early Dementia Detection", page_icon="🧠", layout="centered")


# -------------------------------
# Model loading (cached)
# -------------------------------
@st.cache_resource
def load_model() -> Optional[object]:
    try:
        return joblib.load("mock_hybrid_model.pkl")
    except Exception:
        return None


model = load_model()

# -------------------------------
# Feature extractor
# -------------------------------
def extract_features(filepath: str, label: str = "unknown", task: str = "free") -> dict:
    y, sr = librosa.load(filepath, sr=16000)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Energy-based pause estimate (10th percentile threshold)
    frame_hop = 512
    energy = np.array([np.sum(y[i:i + frame_hop] ** 2) for i in range(0, len(y), frame_hop)])
    pauses = int(np.sum(energy < np.percentile(energy, 10)))

    features = {f"mfcc_{i}": float(mfcc_mean[i]) for i in range(len(mfcc_mean))}
    features.update({
        "pauses": pauses,
        "task": task,
        "label": label,
        "file": filepath,
        "duration_sec": float(len(y) / sr),
    })
    return features


# -------------------------------
# Audio transcription via AssemblyAI
# -------------------------------
def _get_assemblyai_api_key() -> Optional[str]:
    # Prefer Streamlit secrets, fallback to env var
    key = None
    try:
        key = st.secrets.get("ASSEMBLYAI_API_KEY", None)  # type: ignore
    except Exception:
        key = None
    if not key:
        key = os.getenv("ASSEMBLYAI_API_KEY") or os.getenv("AAI_API_KEY")
    return key


def transcribe_audio(file_path: str) -> Tuple[Optional[str], Optional[str]]:
    """Transcribe audio using AssemblyAI if API key is set. Returns (text, error)."""
    api_key = _get_assemblyai_api_key()
    if not api_key or aai is None:
        return None, "AssemblyAI not configured. Set ASSEMBLYAI_API_KEY to enable transcription."

    try:
        # Configure API key
        try:
            # Newer SDKs support settings
            aai.settings.api_key = api_key  # type: ignore[attr-defined]
            transcriber = aai.Transcriber()  # type: ignore
        except Exception:
            # Older SDKs might use Client
            transcriber = aai.Transcriber(aai.Client(api_key=api_key))  # type: ignore

        transcript = transcriber.transcribe(file_path)  # type: ignore
        text = getattr(transcript, "text", None)
        status = getattr(transcript, "status", "completed")
        error = getattr(transcript, "error", None)

        if status == "error" or (text is None and error):
            return None, f"Transcription error: {error or 'Unknown error'}"
        if not text:
            return None, "Transcription returned no text."
        return text, None
    except Exception as exc:  # pragma: no cover - network/runtime dependent
        return None, f"Transcription failed: {exc}"


def _normalize_text(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch not in string.punctuation)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🧠 Early Dementia Detection Prototype")
st.write("Prototype with **speech analysis + cognitive tasks**")

task_type = st.selectbox(
    "Choose a screening task:",
    ["Free Speech (Audio)", 
     "Picture Description (Audio)", 
     "Memory Recall (Audio)", 
     "Verbal Fluency (Audio)", 
     "Word Recall (Text)", 
     "Math Puzzle (Text)", 
     "Category Fluency (Text)"]
)

# -------------------------------
# AUDIO TASKS
# -------------------------------
if "Audio" in task_type:
    uploaded_file = st.file_uploader("Upload your audio file", type=["wav", "mp3", "m4a"])

    if uploaded_file is not None:
        suffix = os.path.splitext(uploaded_file.name or "audio.wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            feat = extract_features(tmp_path, task=task_type)
            feat_df = pd.DataFrame([feat]).drop(columns=["file", "label", "task"])
        except Exception as e:
            feat = {"duration_sec": None, "pauses": None}
            feat_df = None
            st.warning(f"Feature extraction issue: {e}")

        st.subheader("Audio Task Results")
        st.write(f"🧾 **Task:** {task_type}")

        if model is not None and feat_df is not None:
            try:
                pred = model.predict(feat_df)[0]
                proba = model.predict_proba(feat_df)[0]
                st.write(f"🧠 **Prediction:** {pred}")
                st.write(f"📊 **Probabilities:** Healthy = {proba[0]:.2f}, Dementia = {proba[1]:.2f}")
            except Exception as e:
                st.warning(f"Model inference failed: {e}")
        else:
            st.info("Model not available. Skipping prediction.")

        # --- Transcription via API ---
        with st.spinner("Transcribing audio (if API key configured)..."):
            transcript_text, transcript_err = transcribe_audio(tmp_path)

        if transcript_text:
            st.markdown("---")
            st.subheader("Transcript")
            st.write(transcript_text)
        elif transcript_err:
            st.info(transcript_err)

        # --- Task-specific analysis using transcript and/or acoustics ---
        st.markdown("---")
        st.subheader("Task-specific Analysis")

        duration_sec = feat.get("duration_sec") if isinstance(feat, dict) else None
        pauses = feat.get("pauses") if isinstance(feat, dict) else None

        if "Picture" in task_type:
            if transcript_text:
                text = transcript_text
                word_count = len(text.split())
                sentence_count = sum(text.count(x) for x in ".!?") or 1
                st.metric("Word Count", word_count)
                st.metric("Sentence Count", sentence_count)
            st.info("🖼 Imagine describing a picture (like 'Cookie Theft').")

        elif "Memory Recall" in task_type:
            st.info("🧾 Typically, the app would play a story, then ask recall.")
            if transcript_text:
                key_points = [
                    "anna", "florist", "greenville", "tuesday", "blue bus", "42", "10 am",
                    "market", "apples", "silver key", "leo", "musician", "clock tower",
                ]
                normalized = _normalize_text(transcript_text)
                hits = 0
                for kp in key_points:
                    if _normalize_text(kp) in normalized:
                        hits += 1
                st.metric("Recall Score", f"{hits} / {len(key_points)} key details")
                with st.expander("Key points checked"):
                    st.write(", ".join(f"`{kp}`" for kp in key_points))

        elif "Verbal Fluency" in task_type:
            if transcript_text and duration_sec and duration_sec > 0:
                words = [w for w in _normalize_text(transcript_text).split() if w]
                unique_words = set(words)
                wpm = int(len(words) / (duration_sec / 60.0))
                st.metric("Speech Rate", f"{wpm} WPM")
                st.metric("Unique Words", len(unique_words))
            if pauses is not None:
                st.metric("Pause Count (energy-based)", pauses)
            st.info("🔤 Example prompt: 'Name as many animals as possible in 1 minute.'")

        else:  # Free Speech or other audio tasks
            if transcript_text:
                words = transcript_text.split()
                st.metric("Word Count", len(words))
            if duration_sec:
                st.metric("Duration", f"{duration_sec:.1f} sec")
            if pauses is not None:
                st.metric("Pause Count (energy-based)", pauses)

        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

# -------------------------------
# TEXT TASKS
# -------------------------------
elif task_type == "Word Recall (Text)":
    words = ["apple", "river", "chair", "flower", "book"]
    st.write("Memorize these words for 10 seconds:")
    st.write("🔑 " + ", ".join(words))

    if st.button("I am ready to recall"):
        recalled = st.text_area("Type the words you remember:")
        if recalled:
            recalled_list = recalled.lower().split()
            score = sum([1 for w in words if w in recalled_list])
            st.success(f"You recalled {score}/5 words correctly.")

elif task_type == "Math Puzzle (Text)":
    a, b = random.randint(10,99), random.randint(10,99)
    st.write(f"Solve this quickly: {a} + {b} = ?")
    answer = st.number_input("Your Answer:", step=1)
    if answer:
        if answer == (a+b):
            st.success("✅ Correct!")
        else:
            st.error("❌ Incorrect")

elif task_type == "Category Fluency (Text)":
    st.write("Type as many **fruits** as you can in 30 seconds:")
    fruits = st.text_area("Start typing:")
    if fruits:
        count = len(fruits.split())
        st.success(f"You listed {count} fruits.")
        if count < 5:
            st.warning("Low score (possible cognitive decline).")
        else:
            st.info("Good performance.")
