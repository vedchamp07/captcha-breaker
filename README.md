# 🔐 CAPTCHA Breaker

A deep learning project to recognize CAPTCHA images using PyTorch with **CTC (Connectionist Temporal Classification)** architecture.

**No bounding boxes needed** - the model automatically learns character positions!

## 🎯 Project Overview

This project uses a CNN + LSTM + CTC architecture to recognize text in CAPTCHA images. It includes:

- CAPTCHA image generation using the `python-captcha` library
- CTC-based model (industry standard for sequence recognition)
- Preprocessing pipeline (grayscale conversion, noise removal)
- Training and prediction scripts
- Kaggle GPU training support with step-by-step guide

**Why CTC?**

- Handles variable character spacing
- Works with overlapping/distorted characters
- No manual bounding box labeling needed
- Used in production OCR systems (Google Tesseract, etc.)

## 📁 Project Structure

```
captcha-breaker/
├── src/
│   ├── __init__.py
│   └── model.py              # CTC-based CAPTCHA model
├── data/
│   ├── raw/                  # Generated CAPTCHA images
│   └── processed/            # Preprocessed images (grayscale)
├── models/
│   └── captcha_model.pth     # Trained model weights
├── notebooks/
│   └── kaggle_training.ipynb # Kaggle training notebook
├── generate_dataset.py       # Generate synthetic CAPTCHAs
├── preprocess.py             # Preprocess images (grayscale, denoise)
├── train.py                  # Train the CTC model
├── predict.py                # Predict on single image
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── ARCHITECTURE_COMPARISON.md # Explanation of different approaches
└── KAGGLE_WORKFLOW.md        # Complete Kaggle guide
```

## 🚀 Quick Start

### Local Training

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/captcha-breaker.git
cd captcha-breaker

# 2. Install dependencies
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
# venv\Scripts\activate   # On Windows
pip install -r requirements.txt

# 3. Generate dataset
python generate_dataset.py

# 4. Preprocess images
python preprocess.py

# 5. Train the model
python train.py

# 6. Test prediction
python predict.py data/processed/ABC12_0.png
```

### Kaggle Training (Recommended for GPU)

See **[KAGGLE_WORKFLOW.md](KAGGLE_WORKFLOW.md)** for complete step-by-step instructions.

**Quick version:**

1. Upload code to Kaggle as dataset or use the notebook template
2. Use the provided `notebooks/kaggle_training.ipynb`
3. Enable GPU in Kaggle settings
4. Run all cells
5. Download trained model
6. Push to GitHub using Kaggle secrets

## 🏗️ Model Architecture

```
Input Image (60x160 grayscale)
    ↓
CNN Feature Extraction (4 conv blocks)
    ↓
Reshape to Sequence (width → time steps)
    ↓
Bidirectional LSTM (2 layers)
    ↓
Character Predictions (per time step)
    ↓
CTC Loss (automatic alignment)
    ↓
Output: 5 Characters (A-Z, 0-9)
```

**Key Components:**

- **CNN Backbone**: Extracts visual features from CAPTCHA
- **LSTM**: Processes sequential information
- **CTC Loss**: Handles alignment without explicit position labels
- **No Bounding Boxes**: Model learns character positions automatically

See [ARCHITECTURE_COMPARISON.md](ARCHITECTURE_COMPARISON.md) for comparison with other approaches.

## 📊 Performance

| Metric            | Value                 |
| ----------------- | --------------------- |
| Training Time     | 15-30 min (GPU)       |
| Expected Accuracy | 50-90%                |
| Model Size        | ~5-20 MB              |
| Dataset Size      | 10,000 images         |
| Character Set     | 36 classes (0-9, A-Z) |

**Previous approaches:**

- Original simple CNN: ~30-50% accuracy
- Two-stage with bbox: **6.25%** (broken due to incorrect bbox labels)
- **CTC approach (current): 50-90%** ✅

## 💻 Usage

### Generate Dataset

```bash
python generate_dataset.py
```

Creates 10,000 synthetic CAPTCHA images in `data/raw/`

### Preprocess Images

```bash
python preprocess.py
```

Converts to grayscale and removes noise → saves to `data/processed/`

### Train Model

```bash
python train.py
```

Trains for 50 epochs, saves best model to `models/captcha_model.pth`

### Predict

```bash
python predict.py data/processed/ABC12_0.png
```

Outputs: Predicted text and comparison with ground truth (if available)

## 🔧 Configuration

**generate_dataset.py:**

- `NUM_SAMPLES = 10000` - Number of images to generate
- `CAPTCHA_LENGTH = 5` - Length of CAPTCHA text

**train.py:**

- `BATCH_SIZE = 64` - Training batch size
- `EPOCHS = 50` - Number of training epochs
- `LEARNING_RATE = 0.001` - Initial learning rate
- `USE_LSTM = True` - Use LSTM model (set False for simpler CNN-only)

**preprocess.py:**

- Grayscale conversion
- Otsu's thresholding for binarization
- Morphological operations for noise removal

## 📚 Additional Documentation

- **[ARCHITECTURE_COMPARISON.md](ARCHITECTURE_COMPARISON.md)** - Why CTC? Comparison of different approaches
- **[KAGGLE_WORKFLOW.md](KAGGLE_WORKFLOW.md)** - Complete step-by-step Kaggle training guide
- **[notebooks/kaggle_training.ipynb](notebooks/kaggle_training.ipynb)** - Ready-to-use Kaggle notebook

## 🤝 Contributing

Feel free to open issues or submit pull requests!

## 📄 License

MIT License

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- python-captcha library for CAPTCHA generation
- CTC loss implementation based on PyTorch's CTCLoss
