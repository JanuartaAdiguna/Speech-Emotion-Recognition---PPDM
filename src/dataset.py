"""
src/dataset.py

Custom PyTorch Dataset untuk Speech Emotion Recognition berbasis
Mel-Spectrogram 2D. Setiap sampel dikembalikan dalam bentuk tensor
[1, n_mels, time_steps] -> dimensi channel=1 dibutuhkan oleh Conv2D.
"""

import numpy as np
import os
import torch
from torch.utils.data import Dataset

from src.preprocessing import extract_melspectrogram, random_augment


class SEREmotionDataset(Dataset):
    """
    Args:
        file_paths (list[str]) : daftar path file .wav
        labels     (list[int]) : label kelas (integer) untuk setiap file
        train      (bool)      : jika True, augmentasi acak diterapkan
                                  setiap kali sampel diambil (on-the-fly).
                                  Jika False (val/test), tidak ada augmentasi.
        max_len    (int)       : panjang frame waktu tetap
        n_mels     (int)       : jumlah mel bands
    """

    def __init__(self, file_paths, labels, train=True, max_len=200, n_mels=128):
        assert len(file_paths) == len(labels), "Jumlah file_paths dan labels harus sama"
        self.file_paths = file_paths
        self.labels = labels
        self.train = train
        self.max_len = max_len
        self.n_mels = n_mels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # Augmentasi HANYA diterapkan pada mode training, dan dipilih
        # secara acak setiap kali sampel ini diambil (bukan sekali di awal),
        # sehingga model melihat variasi berbeda di setiap epoch.
        augment_fn = random_augment if self.train else None

        mel_db = extract_melspectrogram(
            file_path,
            n_mels=self.n_mels,
            max_len=self.max_len,
            augment_fn=augment_fn,
        )

        if mel_db is None:
            # Fallback: jika file gagal dibaca, kembalikan tensor kosong (nol)
            # agar training tidak crash. Sebaiknya file bermasalah dicek manual.
            mel_db = np.zeros((self.n_mels, self.max_len), dtype=np.float32)

        # Normalisasi per-sampel (zero mean, unit variance).
        # Ini bukan bagian dari requirement eksplisit, tetapi sangat disarankan
        # karena nilai dB mentah bisa bervariasi rentangnya antar file,
        # sehingga training CNN + BatchNorm menjadi lebih stabil.
        mean = mel_db.mean()
        std = mel_db.std() + 1e-6
        mel_db = (mel_db - mean) / std

        # Tambahkan dimensi channel -> [1, n_mels, time_steps]
        feature = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return feature, label_tensor


def load_ravdess_files(data_dir: str):
    """
    Utility: Telusuri `data_dir` secara rekursif dan kembalikan daftar
    file .wav beserta label string sesuai konvensi penamaan RAVDESS.

    Returns:
        (file_paths, label_strs)
    """
    RAVDESS_EMOTION_MAP = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised",
    }

    file_paths = []
    label_strs = []

    for root, _, files in os.walk(data_dir):
        for fname in files:
            if not fname.lower().endswith(".wav"):
                continue
            parts = fname.split("-")
            if len(parts) < 3 or parts[2] not in RAVDESS_EMOTION_MAP:
                continue
            file_paths.append(os.path.join(root, fname))
            label_strs.append(RAVDESS_EMOTION_MAP[parts[2]])

    return file_paths, label_strs
