"""
src/model.py

Arsitektur Deep 2D CNN untuk Speech Emotion Recognition berbasis
Mel-Spectrogram. Terdiri dari 4 blok konvolusi:
    Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
diikuti oleh fully-connected layers dengan Dropout untuk regularisasi.
"""

import torch
import torch.nn as nn


class EmotionCNN(nn.Module):
    def __init__(self, num_classes: int = 8, n_mels: int = 128, max_len: int = 200, dropout: float = 0.5):
        """
        Args:
            num_classes : jumlah kelas emosi output (default 8)
            n_mels      : tinggi input (jumlah mel bands), harus sama dengan
                          n_mels yang dipakai saat preprocessing
            max_len     : lebar input (jumlah frame waktu)
            dropout     : probabilitas dropout pada FC layers
        """
        super(EmotionCNN, self).__init__()

        # ---- Blok 1: 1 -> 32 channel ----
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # ---- Blok 2: 32 -> 64 channel ----
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # ---- Blok 3: 64 -> 128 channel ----
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # ---- Blok 4: 128 -> 256 channel ----
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # Hitung ukuran flatten SECARA DINAMIS dengan forward pass dummy.
        # Ini penting agar kode tidak mudah error saat n_mels/max_len berubah
        # (hardcoding angka di sini adalah sumber bug paling umum pada CNN audio).
        self._flatten_size = self._compute_flatten_size(n_mels, max_len)

        self.classifier = nn.Sequential(
            nn.Linear(self._flatten_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def _compute_flatten_size(self, n_mels: int, max_len: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_mels, max_len)
            out = self._forward_conv(dummy)
            return int(out.view(1, -1).size(1))

    def _forward_conv(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        return x

    def forward(self, x):
        """
        Input  : x shape [batch, 1, n_mels, time_steps]
        Output : logits shape [batch, num_classes]
        """
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Quick sanity check
    model = EmotionCNN(num_classes=8, n_mels=128, max_len=200)
    dummy_input = torch.randn(4, 1, 128, 200)  # batch=4
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Expected: [4, 8]
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameter: {total_params:,}")
