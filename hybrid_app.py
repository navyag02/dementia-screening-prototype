import streamlit as st
import librosa
import numpy as np
import pandas as pd
import joblib
import tempfile
import random

# Load trained model
model = joblib.load("mock_hybrid_model.pkl")

# -------------------------------
# Feature extractor
# -------------------------------
def extract_features(filepath, label="unknown", task="free"):
    y, sr = librosa.load(filepath, sr=16000)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    energy = np.array([sum(abs(y[i:i+512]**2)) for i in range(0, len(y), 512)])
    pauses = np.sum(energy < np.percentile(energy, 10))

    features = {f"mfcc_{i}": mfcc_mean[i] for i in range(len(mfcc_mean))}
    features.update({
        "pauses": pauses,
        "task": task,
        "label": label,
        "file": filepath
    })
    return features

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
    uploaded_file = st.file_uploader("Upload your .wav file", type=["wav"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        feat = extract_features(tmp_path, task=task_type)
        feat_df = pd.DataFrame([feat]).drop(columns=["file","label","task"])

        pred = model.predict(feat_df)[0]
        proba = model.predict_proba(feat_df)[0]

        st.subheader("Audio Task Results")
        st.write(f"🧾 **Task:** {task_type}")
        st.write(f"🧠 **Prediction:** {pred}")
        st.write(f"📊 **Probabilities:** Healthy = {proba[0]:.2f}, Dementia = {proba[1]:.2f}")

        if "Picture" in task_type:
            st.info("🖼 Imagine describing a picture (like 'Cookie Theft').")
        elif "Memory Recall" in task_type:
            st.info("🧾 Normally, the app would play a story, then ask recall.")
        elif "Verbal Fluency" in task_type:
            st.info("🔤 Example: 'Name as many animals as possible in 1 minute.'")

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
