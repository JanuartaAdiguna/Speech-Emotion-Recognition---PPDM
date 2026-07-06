import os
import sys
import glob
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Import project modules
from src.dataset import load_ravdess_files, SEREmotionDataset
from src.model import EmotionCNN


def evaluate(data_dir: str, model_path: str, batch_size: int = 32, n_mels: int = 128, max_len: int = 200):
    # Load dataset files and string labels using the same RAVDESS loader
    file_paths, label_strs = load_ravdess_files(data_dir)
    if len(file_paths) == 0:
        raise RuntimeError(f"Tidak ada file .wav yang cocok ditemukan di '{data_dir}'.")

    # Ensure consistent label ordering (same as training: sorted unique)
    emotion_labels = sorted(list(set(label_strs)))
    label_to_idx = {label: i for i, label in enumerate(emotion_labels)}
    labels = [label_to_idx[l] for l in label_strs]

    # Final split: reserve 20% for test (same as train.py)
    _, X_test, _, y_test = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    test_dataset = SEREmotionDataset(X_test, y_test, train=False, max_len=max_len, n_mels=n_mels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EmotionCNN(num_classes=len(emotion_labels), n_mels=n_mels, max_len=max_len)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            outputs = model(features)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(targets.numpy().tolist())

    print("\n✅ Classification Report:\n")
    # Map indices back to label names for reporting
    print(classification_report(all_labels, all_preds, target_names=emotion_labels))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=emotion_labels, yticklabels=emotion_labels, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate EmotionCNN on RAVDESS test split')
    parser.add_argument('--data_dir', type=str, default='data/raw_audio', help='Path to dataset folder')
    parser.add_argument('--model_path', type=str, default='emotion_cnn.pth', help='Path to saved model (.pth)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_mels', type=int, default=128)
    parser.add_argument('--max_len', type=int, default=200)
    args = parser.parse_args()

    evaluate(args.data_dir, args.model_path, args.batch_size, args.n_mels, args.max_len)
