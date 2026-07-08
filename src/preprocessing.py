

import numpy as np
import librosa


SAMPLE_RATE = 22050     
N_MELS = 128            
N_FFT = 2048            
HOP_LENGTH = 512        
MAX_LEN = 200           


# ==========================================================
# 1. Fungsi Padding / Truncation
# ==========================================================
def pad_or_truncate(mel_db: np.ndarray, max_len: int = MAX_LEN) -> np.ndarray:
   
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
    
    noise = np.random.randn(len(y))
    augmented = y + noise_factor * noise
    return augmented.astype(np.float32)


def pitch_shift(y: np.ndarray, sr: int, n_steps: float = None) -> np.ndarray:
    
    if n_steps is None:
        n_steps = np.random.uniform(-2, 2)
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps).astype(np.float32)


def time_stretch(y: np.ndarray, sr: int, rate: float = None) -> np.ndarray:
  
    if rate is None:
        rate = np.random.uniform(0.8, 1.2)
    try:
        y_stretched = librosa.effects.time_stretch(y=y, rate=rate)
    except Exception:
        
        y_stretched = y
    return y_stretched.astype(np.float32)


def random_augment(y: np.ndarray, sr: int) -> np.ndarray:
    
    choice = np.random.choice(["none", "noise", "pitch", "stretch"], p=[0.4, 0.2, 0.2, 0.2])

    if choice == "noise":
        y = add_white_noise(y, sr)
    elif choice == "pitch":
        y = pitch_shift(y, sr)
    elif choice == "stretch":
        y = time_stretch(y, sr)

    return y



def extract_melspectrogram(
    file_path: str,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    max_len: int = MAX_LEN,
    augment_fn=None,
):
 
    try:
        y, sr = librosa.load(file_path, sr=sr)
    except Exception as e:
        print(f"[preprocessing] Gagal memuat file '{file_path}': {e}")
        return None

    
    if augment_fn is not None:
        y = augment_fn(y, sr)

    
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=N_FFT, hop_length=HOP_LENGTH
    )

    
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

   
    mel_db = pad_or_truncate(mel_db, max_len=max_len)

    return mel_db.astype(np.float32)
