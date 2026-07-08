
import os
import sys
import tempfile

import streamlit as st
import torch


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import EmotionCNN
from src.preprocessing import extract_melspectrogram


emotion_labels = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]

N_MELS = 128
MAX_LEN = 200
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emotion_cnn.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def render_css():
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
        }
        .stApp {
            background: linear-gradient(135deg, #07111f 0%, #111827 45%, #172554 100%);
        }
        .hero-card {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 24px;
            padding: 24px;
            box-shadow: 0 18px 45px rgba(0,0,0,0.22);
            backdrop-filter: blur(12px);
            margin-bottom: 16px;
        }
        .info-card {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 18px;
            padding: 16px 18px;
            margin-bottom: 12px;
        }
        .result-card {
            background: linear-gradient(135deg, rgba(56,189,248,0.18), rgba(129,140,248,0.20));
            border: 1px solid rgba(255,255,255,0.16);
            border-radius: 22px;
            padding: 20px;
            margin: 10px 0 16px 0;
            box-shadow: 0 12px 30px rgba(15,23,42,0.20);
        }
        div[data-testid="stFileUploader"] > div {
            background: rgba(255,255,255,0.04);
            border: 1px dashed rgba(255,255,255,0.20);
            border-radius: 16px;
            padding: 10px;
        }
        div[data-testid="stExpander"] {
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 14px;
            background: rgba(255,255,255,0.04);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


st.set_page_config(page_title="Emotion Voice Analyzer", page_icon="🎧", layout="wide")
render_css()

st.markdown(
    """
    <div class="hero-card">
        <h1 style="margin:0 0 8px 0; font-size:2rem;">Emotion Voice Analyzer</h1>
        <p style="margin:0; color:#dbeafe; font-size:1rem;">
            Unggah rekaman suara untuk melihat emosi yang paling terasa dari nada bicara Anda.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns([1.35, 0.95], gap="large")

with col1:
    st.markdown("### Upload sample audio")
    st.write("Format yang didukung: .wav. Rekaman dengan kualitas yang jelas akan memberi hasil yang lebih konsisten.")
    uploaded_file = st.file_uploader("Pilih file suara", type=["wav"], label_visibility="visible")

with col2:
    st.markdown(
        """
        <div class="info-card">
            <h3 style="margin-top:0;">Yang bisa Anda lihat</h3>
            <ul style="padding-left: 18px; margin:0; line-height:1.7; color:#e2e8f0;">
                <li>Prediksi emosi seperti marah, tenang, takut, bahagia, dan lainnya.</li>
                <li>Nilai keyakinan untuk hasil yang lebih mudah dibaca.</li>
                <li>Detail probabilitas tiap kelas secara ringkas.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        f"File model '{MODEL_PATH}' tidak ditemukan. "
        "Jalankan `python train.py` terlebih dahulu untuk menghasilkan model."
    )
    st.stop()

if uploaded_file is None:
    st.info("Unggah file audio untuk memulai analisis.")
else:
    with st.spinner("Menganalisis suara..."):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        mel_db = extract_melspectrogram(temp_path, n_mels=N_MELS, max_len=MAX_LEN)

        if mel_db is None:
            st.error("Tidak dapat mengekstrak fitur dari audio yang dikirim.")
        else:
            
            mean = mel_db.mean()
            std = mel_db.std() + 1e-6
            mel_db_norm = (mel_db - mean) / std

           
            x = torch.tensor(mel_db_norm, dtype=torch.float32)
            x = x.unsqueeze(0).unsqueeze(0).to(device)  
            with torch.no_grad():
                output = model(x)
                probs = torch.softmax(output, dim=1)
                predicted_idx = torch.argmax(probs, dim=1).item()
                confidence = probs[0, predicted_idx].item()

            predicted_label = emotion_labels[predicted_idx]
            confidence_pct = int(round(confidence * 100))
            label_display = predicted_label.replace("_", " ").title()

            st.markdown(
                f"""
                <div class="result-card">
                    <div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap;">
                        <div style="font-size:2rem;">🎧</div>
                        <div>
                            <h3 style="margin:0;">{label_display}</h3>
                            <p style="margin:4px 0 0 0; color:#e2e8f0;">Keyakinan hasil: {confidence_pct}%</p>
                        </div>
                    </div>
                    <div style="margin-top:14px;">
                        <p style="margin:0 0 6px 0; font-weight:600;">Confidence</p>
                        <div style="height:10px; background:rgba(255,255,255,0.12); border-radius:999px; overflow:hidden;">
                            <div style="height:100%; width:{confidence_pct}%; background:linear-gradient(90deg,#38bdf8,#818cf8); border-radius:999px;"></div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("Lihat detail probabilitas tiap kelas"):
                for label, p in zip(emotion_labels, probs[0].tolist()):
                    st.progress(p, text=f"{label.replace('_', ' ').title()}: {p * 100:.1f}%")

            st.audio(temp_path)

    try:
        os.remove(temp_path)
    except OSError:
        pass
