"""
train.py

Script training untuk EmotionCNN (Deep 2D CNN berbasis Mel-Spectrogram).

ASUMSI DATASET:
Karena daftar `emotion_labels` yang Anda berikan
    ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
persis sama dengan 8 kelas emosi pada dataset RAVDESS (diurutkan alfabetis),
script ini diasumsikan menggunakan struktur penamaan file RAVDESS, contoh:
    03-01-06-01-02-01-12.wav
    (kode emosi ada di posisi ke-3: 06 = fearful)

Jika Anda menggunakan dataset lain (mis. CREMA-D, atau folder per-kelas seperti
    data/angry/xxx.wav, data/happy/yyy.wav, dst.),
CUKUP GANTI fungsi `gather_dataset_files()` di bawah ini sesuai struktur folder Anda.
Bagian training loop TIDAK perlu diubah.

Cara menjalankan:
    python train.py --data_dir data/RAVDESS --epochs 50
"""

import os
import copy
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.dataset import SEREmotionDataset, load_ravdess_files
from src.model import EmotionCNN

# Mapping kode emosi RAVDESS (posisi ke-3 pada nama file) -> nama emosi
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


# Dataset discovery is provided by `load_ravdess_files()` in `src/dataset.py`.


def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device}")

    file_paths, label_strs = load_ravdess_files(args.data_dir)
    if len(file_paths) == 0:
        raise RuntimeError(
            f"Tidak ada file .wav yang cocok ditemukan di '{args.data_dir}'.\n"
            "Periksa kembali path dataset, atau sesuaikan fungsi "
            "gather_dataset_files() di train.py dengan struktur dataset Anda."
        )

    # Urutkan label secara alfabetis agar konsisten dengan app.py
    emotion_labels = sorted(list(set(label_strs)))
    label_to_idx = {label: i for i, label in enumerate(emotion_labels)}
    labels = [label_to_idx[l] for l in label_strs]

    print(f"Total sampel ditemukan : {len(file_paths)}")
    print(f"Mapping label -> index : {label_to_idx}")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_dataset = SEREmotionDataset(
        train_paths, train_labels, train=True, max_len=args.max_len, n_mels=args.n_mels
    )
    val_dataset = SEREmotionDataset(
        val_paths, val_labels, train=False, max_len=args.max_len, n_mels=args.n_mels
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = EmotionCNN(
        num_classes=len(emotion_labels), n_mels=args.n_mels, max_len=args.max_len
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # ReduceLROnPlateau: turunkan learning rate saat val accuracy stagnan
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        # ---------------- Training ----------------
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # ---------------- Validation ----------------
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)

                val_running_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total

        # Scheduler dipantau berdasarkan val accuracy (mode='max')
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}"
        )

        # ---------------- Checkpoint + Early Stopping ----------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            torch.save(best_model_wts, args.save_path)
            print(f"  -> Model terbaik disimpan (Val Acc: {best_val_acc:.4f}) ke '{args.save_path}'")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(
                    f"Early stopping dipicu pada epoch {epoch} "
                    f"(tidak ada peningkatan selama {args.patience} epoch berturut-turut)."
                )
                break

    print(f"\nTraining selesai. Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Model terbaik tersimpan di: {args.save_path}")
    print(f"PENTING - urutan label ini HARUS sama persis dengan emotion_labels di app.py:")
    print(f"  {emotion_labels}")


def parse_args():
    parser = argparse.ArgumentParser(description="Training EmotionCNN (Mel-Spectrogram 2D CNN)")
    parser.add_argument("--data_dir", type=str, default="data/RAVDESS", help="Path ke folder dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--patience", type=int, default=8, help="Jumlah epoch tanpa perbaikan sebelum early stop")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of DataLoader worker processes (0 to disable multiprocessing on Windows)")
    parser.add_argument("--save_path", type=str, default="emotion_cnn.pth")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
