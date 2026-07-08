# Speech Emotion Recognition — Deep 2D CNN (Mel-Spectrogram)


## Struktur Project

```
ser_project/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py   # Ekstraksi Mel-Spectrogram + augmentasi
│   ├── dataset.py          # Custom PyTorch Dataset
│   └── model.py             # EmotionCNN (Deep 2D CNN)
├── train.py                  # Script training
├── app.py                    # Aplikasi Streamlit untuk inference
├── requirements.txt
└── README.md
```

## Instalasi

```bash
python -m venv venv

pip install -r requirements.txt
```

## Dataset

Script `train.py` secara default mengasumsikan dataset **RAVDESS**
(karena 8 label emosi Anda cocok persis dengan RAVDESS), dengan format nama file:

```
03-01-06-01-02-01-12.wav
      ^^ kode emosi (posisi ke-3): 01=neutral, 02=calm, 03=happy,
                                    04=sad, 05=angry, 06=fearful,
                                    07=disgust, 08=surprised
```

Letakkan seluruh file `.wav` (boleh di dalam sub-folder Actor_01, Actor_02, dst.)
di dalam satu folder, misalnya `data/RAVDESS/`.

**Jika dataset Anda BUKAN RAVDESS**, edit fungsi `gather_dataset_files()`
di `train.py` — bagian training loop tidak perlu diubah sama sekali.


## Usage — Train, Evaluate, Run App




- **Training (default menyimpan model terbaik sebagai `emotion_cnn.pth`)**

```bash
# training penuh (contoh 50 epoch)
python train.py --data_dir data/raw_audio --epochs 50 --batch_size 32 --save_path emotion_cnn.pth

# quick smoke test (1 epoch)
python train.py --data_dir data/raw_audio --epochs 1 --batch_size 8 --save_path tmp_emotion_cnn.pth
```

Script training otomatis:
- menemukan file .wav menggunakan loader RAVDESS di [src/dataset.py](src/dataset.py#L1)
- melakukan split train/val (80/20), training loop, checkpoint model terbaik berdasarkan validation accuracy


- **Evaluasi**

Gunakan [evaluate.py](evaluate.py#L1) untuk menjalankan evaluasi pada test split dan menampilkan confusion matrix.

```bash
python evaluate.py --data_dir data/raw_audio --model_path emotion_cnn.pth --batch_size 32
```

Jika ingin menilai model sementara hasil training singkat, arahkan `--model_path` ke file yang dihasilkan (mis. `tmp_emotion_cnn.pth`).


- **Menjalankan web app untuk inference**

```bash
streamlit run app.py
```

Pastikan parameter `n_mels` dan `max_len` di `app.py` sama dengan saat training (default: 128 dan 200). Jika Anda mengubah konfigurasi training, sesuaikan juga di `app.py`.

--
