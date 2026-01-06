# 🔐 CAPTCHA Breaker

A deep learning project to recognize CAPTCHA images using PyTorch with **CTC (Connectionist Temporal Classification)** architecture.

**No bounding boxes needed** — the model automatically learns character positions!

## 🎯 Overview

This project uses a **CNN + LSTM + CTC** architecture to recognize text in CAPTCHA images. CTC is the industry-standard approach for sequence recognition without explicit alignment labels.

### Why CTC?

- ✅ Handles variable character spacing and positions
- ✅ Works with overlapping/distorted characters
- ✅ No manual bounding box labeling needed
- ✅ Used in production OCR systems (Google Tesseract, etc.)

## 📁 Project Structure

```
captcha-breaker/
├── src/
│   ├── __init__.py
│   └── model.py                    # CTC-based CAPTCHA model
├── data/
│   ├── raw/                        # Generated CAPTCHA images
│   └── processed/                  # Preprocessed images (grayscale)
├── models/
│   └── captcha_model.pth           # Trained model weights
├── generate_dataset.py             # Generate synthetic CAPTCHAs
├── preprocess.py                   # Preprocess images (grayscale, denoise)
├── train.py                        # Train the CTC model
├── predict.py                      # Predict on single image
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Quick Start

### Local Setup

```bash
# 1. Clone repository
git clone https://github.com/vedchamp07/captcha-breaker.git
cd captcha-breaker

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate dataset
python generate_dataset.py

# 5. Preprocess images
python preprocess.py

# 6. Train model
python train.py

# 7. Test prediction
python predict.py data/processed/ABC12_0.png
```

### Kaggle GPU Training

Train on Kaggle with GPU (works for private repos):

1. Enable GPU in notebook settings.
2. Clone the repo using a GitHub token stored in Kaggle Secrets, then run training.

```python
from kaggle_secrets import UserSecretsClient
import os

user_secrets = UserSecretsClient()
token = user_secrets.get_secret("GITHUB_TOKEN")
os.system(f"git clone https://{token}@github.com/vedchamp07/captcha-breaker.git /kaggle/working/captcha-breaker")
os.chdir("/kaggle/working/captcha-breaker")
```

```bash
pip install -r requirements.txt
python generate_dataset.py
python preprocess.py
python train.py
```

## 🏗️ Model Architecture

```
Input Image (1, 60, 160)
    ↓
CNN: 4 Convolutional Blocks
  • Progressively extract features: 1→32→64→128→256
  • BatchNorm + ReLU activation
  • Strategic MaxPooling (height & width → width only)
    ↓
Sequence Reshaping: (256, 15, 10) → (batch, 10, 3840)
  • Treat width dimension (10) as time steps
  • Flatten height×channels (15×256=3840) as features
    ↓
Bidirectional LSTM (2 layers, 256 hidden units)
  • Forward + Backward context
  • Outputs: (batch, 10, 512)
    ↓
Linear Classifier: 512 → 37 outputs
  • 36 character classes (0-9, A-Z)
  • +1 for CTC blank token
    ↓
CTC Loss: Automatic alignment learning
    ↓
Greedy Decoding: Argmax + blank/duplicate removal
    ↓
Output: 5-character sequence
```

**Model Stats:**

- Parameters: ~4.02M
- Input: Grayscale 60×160 images
- Output: 5 characters from {0-9, A-Z}

## 💻 Usage

### Generate Synthetic Dataset

```bash
python generate_dataset.py
```

Creates 10,000 random 5-character CAPTCHAs in `data/raw/`

### Preprocess Images

```bash
python preprocess.py
```

Converts to grayscale, applies denoising → saves to `data/processed/`

### Train Model

```bash
python train.py
```

Trains for 50 epochs with:

- Batch size: 64
- Learning rate: 0.001 (with ReduceLROnPlateau scheduler)
- CTC loss with automatic alignment
- Best model saved to `models/captcha_model.pth`

### Make Predictions

```bash
python predict.py <image_path>
```

Example:

```bash
python predict.py data/processed/ABC12_0.png
```

Output:

```
Predicted: ABC12
Ground Truth: ABC12
Correct: ✓
```

## ⚙️ Configuration

Edit these scripts to customize:

**`generate_dataset.py`**

```python
NUM_SAMPLES = 10000      # Number of images
CAPTCHA_LENGTH = 5       # Characters per CAPTCHA
```

**`train.py`**

```python
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
USE_LSTM = True          # Set False for CNN-only model
```

**`preprocess.py`**

- Grayscale conversion
- Otsu's thresholding
- Morphological noise removal

## 📊 Performance

| Metric        | Value           |
| ------------- | --------------- |
| Accuracy      | 50-90%          |
| Training Time | 15-30 min (GPU) |
| Model Size    | ~5-20 MB        |
| Dataset       | 10,000 images   |
| Classes       | 36 (0-9, A-Z)   |

## 🛠️ Technology Stack

- **PyTorch 2.0+** - Deep learning framework
- **torchvision** - Image processing
- **python-captcha** - CAPTCHA generation
- **Pillow** - Image manipulation
- **OpenCV** - Advanced image processing
- **NumPy** - Array operations

## 📝 Notes

- Images are 60×160 grayscale
- 5-character CAPTCHA: digits (0-9) + uppercase letters (A-Z)
- CTC handles variable character spacing without explicit bounding boxes
- Model works on CPU but trains much faster on GPU (50× speedup typical)

## 🤝 Contributing

Issues and pull requests welcome!

## 📄 License

MIT License

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- python-captcha library for CAPTCHA generation
- CTC loss concept from Graves et al. (2006)
