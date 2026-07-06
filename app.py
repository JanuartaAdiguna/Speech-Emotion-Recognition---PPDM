"""
app.py

Aplikasi Streamlit untuk Speech Emotion Recognition,
menggunakan EmotionCNN (Deep 2D CNN) + Mel-Spectrogram.
"""

import os
import sys

import streamlit as st
import torch

# Tambahkan root folder project ke sys.path agar import "src.xxx" berhasil
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import EmotionCNN
from src.preprocessing import extract_melspectrogram

# Urutan label INI HARUS SAMA PERSIS dengan urutan label saat training
# (dicetak otomatis oleh train.py di akhir proses training).
emotion_labels = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]

N_MELS = 128
MAX_LEN = 200
MODEL_PATH = "emotion_cnn.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model():
    """
    Load model sekali saja dan simpan di cache Streamlit,
    supaya tidak perlu load ulang setiap kali user upload file.
    """
    model = EmotionCNN(num_classes=len(emotion_labels), n_mels=N_MELS, max_len=MAX_LEN)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


st.set_page_config(page_title="🎙️ Emotion from Voice", layout="centered")
st.title("🎙️ Speech Emotion Recognition App")
st.write("Upload a `.wav` file and let the AI guess the emotion!")

# Load model dengan penanganan error jika file .pth belum ada
try:
    model = load_model()
except FileNotFoundError:
    st.error(
        f"File model '{MODEL_PATH}' tidak ditemukan. "
        "Jalankan `python train.py` terlebih dahulu untuk menghasilkan model."
    )
    st.stop()

uploaded_file = st.file_uploader("Choose a .wav file", type="wav")

if uploaded_file is not None:
    temp_path = "temp.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    mel_db = extract_melspectrogram(temp_path, n_mels=N_MELS, max_len=MAX_LEN)

    if mel_db is None:
        st.error("Error extracting features from the audio.")
    else:
        # Normalisasi per-sampel, HARUS SAMA dengan yang dilakukan di dataset.py
        # saat training, agar distribusi input konsisten (train vs inference).
        mean = mel_db.mean()
        std = mel_db.std() + 1e-6
        mel_db_norm = (mel_db - mean) / std

        # Bentuk tensor akhir: [batch=1, channel=1, n_mels=128, time_steps]
        x = torch.tensor(mel_db_norm, dtype=torch.float32)
        x = x.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 128, time_steps]

        with torch.no_grad():
            output = model(x)
            probs = torch.softmax(output, dim=1)
            predicted_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, predicted_idx].item()

        st.success(
            f"🎧 Predicted Emotion: **{emotion_labels[predicted_idx]}** "
            f"({confidence * 100:.1f}% confidence)"
        )

        # Tampilkan seluruh probabilitas kelas (opsional, membantu debugging)
        with st.expander("Lihat detail probabilitas semua kelas"):
            for label, p in zip(emotion_labels, probs[0].tolist()):
                st.write(f"{label}: {p * 100:.2f}%")

        st.audio(temp_path)

    os.remove(temp_path)
