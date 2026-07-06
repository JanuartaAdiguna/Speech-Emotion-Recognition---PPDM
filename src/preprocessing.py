"""
src/preprocessing.py

Modul ini bertanggung jawab untuk:
1. Mengekstrak fitur Mel-Spectrogram (2D) dari file audio menggunakan librosa.
2. Menyediakan fungsi augmentasi data (White Noise, Pitch Shift, Time Stretch)
   yang diterapkan HANYA pada data training untuk mencegah overfitting.
3. Melakukan padding/truncation agar semua sampel memiliki panjang waktu
   (time frame) yang seragam, sehingga bisa di-batch oleh DataLoader.
"""

import numpy as np
import librosa

# ==========================================================
# Konfigurasi global (disesuaikan dengan kebutuhan Conv2D)
# ==========================================================
SAMPLE_RATE = 22050     # sample rate standar librosa
N_MELS = 128            # jumlah mel bands -> menjadi tinggi (Height) gambar spektrogram
N_FFT = 2048            # ukuran window FFT
HOP_LENGTH = 512        # langkah antar frame -> menentukan resolusi waktu
MAX_LEN = 200           # panjang frame waktu tetap (Width), sesuai requirement


# ==========================================================
# 1. Fungsi Padding / Truncation
# ==========================================================
def pad_or_truncate(mel_db: np.ndarray, max_len: int = MAX_LEN) -> np.ndarray:
    """
    Menyamakan panjang axis waktu (kolom) dari spektrogram menjadi max_len.
    - Jika lebih panjang -> dipotong (truncate).
    - Jika lebih pendek  -> di-pad dengan nol di sisi kanan.
    """
    current_len = mel_db.shape[1]

    if current_len > max_len:
        mel_db = mel_db[:, :max_len]
    elif current_len < max_len:
        pad_width = max_len - current_len
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant")

    return mel_db


# ==========================================================
# 2. Fungsi Augmentasi Data (bekerja pada sinyal waveform mentah)
# ==========================================================
def add_white_noise(y: np.ndarray, sr: int, noise_factor: float = 0.005) -> np.ndarray:
    """Menambahkan white noise acak ke sinyal audio."""
    noise = np.random.randn(len(y))
    augmented = y + noise_factor * noise
    return augmented.astype(np.float32)


def pitch_shift(y: np.ndarray, sr: int, n_steps: float = None) -> np.ndarray:
    """Menggeser pitch audio secara acak antara -2 hingga +2 semitone."""
    if n_steps is None:
        n_steps = np.random.uniform(-2, 2)
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps).astype(np.float32)


def time_stretch(y: np.ndarray, sr: int, rate: float = None) -> np.ndarray:
    """
    Mempercepat/memperlambat audio secara acak (rate antara 0.8x - 1.2x).
    Catatan: proses ini mengubah panjang sinyal, tetapi tidak masalah karena
    pad_or_truncate() akan menormalkan panjangnya kembali setelah spektrogram dihitung.
    """
    if rate is None:
        rate = np.random.uniform(0.8, 1.2)
    try:
        y_stretched = librosa.effects.time_stretch(y=y, rate=rate)
    except Exception:
        # Fallback jika sinyal terlalu pendek untuk di-stretch
        y_stretched = y
    return y_stretched.astype(np.float32)


def random_augment(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Memilih SATU jenis augmentasi secara acak (atau tidak sama sekali)
    untuk setiap sampel training. Ini mencegah augmentasi menjadi terlalu
    agresif jika ditumpuk semua sekaligus.
    """
    choice = np.random.choice(["none", "noise", "pitch", "stretch"], p=[0.4, 0.2, 0.2, 0.2])

    if choice == "noise":
        y = add_white_noise(y, sr)
    elif choice == "pitch":
        y = pitch_shift(y, sr)
    elif choice == "stretch":
        y = time_stretch(y, sr)

    return y


# ==========================================================
# 3. Fungsi Utama: Ekstraksi Mel-Spectrogram
# ==========================================================
def extract_melspectrogram(
    file_path: str,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    max_len: int = MAX_LEN,
    augment_fn=None,
):
    """
    Mengekstrak Mel-Spectrogram (dalam skala dB) dari sebuah file audio.

    Args:
        file_path : path menuju file .wav
        sr        : target sample rate
        n_mels    : jumlah mel bands (Height gambar spektrogram)
        max_len   : jumlah frame waktu tetap (Width gambar spektrogram)
        augment_fn: fungsi augmentasi opsional, contohnya `random_augment`.
                    Hanya diisi saat memproses data TRAINING.

    Returns:
        np.ndarray dengan shape (n_mels, max_len), atau None jika gagal load.
    """
    try:
        y, sr = librosa.load(file_path, sr=sr)
    except Exception as e:
        print(f"[preprocessing] Gagal memuat file '{file_path}': {e}")
        return None

    # Terapkan augmentasi pada waveform mentah (jika ada)
    if augment_fn is not None:
        y = augment_fn(y, sr)

    # Hitung Mel-Spectrogram (power spectrogram)
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=N_FFT, hop_length=HOP_LENGTH
    )

    # Konversi dari power scale ke decibel (dB) scale, sesuai requirement
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Samakan panjang waktu agar konsisten di semua sampel
    mel_db = pad_or_truncate(mel_db, max_len=max_len)

    return mel_db.astype(np.float32)
