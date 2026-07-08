import numpy as np
import os
import torch
from torch.utils.data import Dataset

from src.preprocessing import extract_melspectrogram, random_augment


class SEREmotionDataset(Dataset):
 

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

        
        augment_fn = random_augment if self.train else None

        mel_db = extract_melspectrogram(
            file_path,
            n_mels=self.n_mels,
            max_len=self.max_len,
            augment_fn=augment_fn,
        )

        if mel_db is None:
            
            mel_db = np.zeros((self.n_mels, self.max_len), dtype=np.float32)

        
        mean = mel_db.mean()
        std = mel_db.std() + 1e-6
        mel_db = (mel_db - mean) / std

       
        feature = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return feature, label_tensor


def load_ravdess_files(data_dir: str):
    
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


def extract_actor_id(file_path: str) -> str:
    
    fname = os.path.basename(file_path)
    parts = fname.split("-")
    if len(parts) < 7:
        return "unknown"
    actor_part = parts[6]
    actor_id = os.path.splitext(actor_part)[0]
    return actor_id