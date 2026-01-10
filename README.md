# 🔐 CAPTCHA Breaker

A deep learning project to recognize CAPTCHA images using PyTorch with **CTC (Connectionist Temporal Classification)** architecture.

**No bounding boxes needed** — the model automatically learns character positions!

### 🌐 **[Try Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/vedchamp07/break-captcha)** ← Click here to test it!

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
│   └── model.py                         # CTC-based CAPTCHA model
├── data/
│   ├── train/                           # Training dataset (gitignored)
│   │   └── raw/                         # Generated CAPTCHA images
│   ├── font_test/                       # Font test dataset (gitignored)
│   │   └── raw/
│   ├── test/                            # Test dataset (gitignored)
│   │   └── raw/
│   └── metadata.json                    # Dataset documentation
├── train_font_library/                  # Custom fonts (download separately)
├── models/
│   └── captcha_model_v4.pth             # Trained model weights (latest)
├── generate_dataset.py                  # Generate synthetic CAPTCHAs (confusion-aware)
├── generate_all_datasets.py             # Generate all datasets at once
├── train.py                             # Train the CTC model
├── predict.py                           # Predict on single image
├── evaluate.py                          # Batch evaluation with metrics
├── analyze_confusion.py                 # Analyze character confusion matrix
├── app.py                               # Gradio web interface
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Download Font Library** (optional, for font testing):
   - Download fonts from [Google Drive](https://drive.google.com/drive/folders/1mv5Z4Z8xpMHryvfceCBbgtObwcVjGApI)
   - Extract to `train_font_library/` directory

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

# 4. Generate training dataset (with confusion-aware sampling)
python generate_dataset.py

# 5. Train model (preprocessing happens on-the-fly)
python train.py

# 6. Test prediction
python predict.py data/train/raw/abc123_0.png --use-attention

# 7. Batch evaluate
python evaluate.py --model models/captcha_model_v4.pth --data-dir data/test/raw

# 8. Analyze character confusion (0/O, 1/I/l, etc.)
python analyze_confusion.py --model models/captcha_model_v4.pth
```

### Using the Gradio App

```bash
python app.py
# Opens interactive web interface at http://localhost:7860
```

### Kaggle GPU Training

Train on Kaggle with GPU (recommended for speed):

1. Enable GPU in notebook settings.
2. Clone the repo using a GitHub token stored in Kaggle Secrets.

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
  • Optional 4-head self-attention refinement (enabled in training script)
    ↓
Linear Classifier: 512 → 37 outputs
  • 36 character classes (0-9, A-Z)
  • +1 for CTC blank token
    ↓
CTC Loss: Automatic alignment learning
    ↓
Greedy Decoding: Argmax + blank/duplicate removal
    ↓
Output: Variable-length prediction (3-7 characters)
```

**Model Stats:**

- Parameters: ~4.02M
- Input: Grayscale 60×160 images
- Output: 3-7 characters from {0-9, a-z, A-Z}
- Character Set: 62 total (digits + lowercase + uppercase)

## 💻 Usage

### Generate Synthetic Datasets

```bash
# Training dataset (variable-length, 3-7 chars)
python generate_dataset.py

# Generates: data/train/raw/ with ~10,000 CAPTCHAs
```

### Preprocessing

**No longer needed!** Preprocessing now happens automatically:

- Grayscale conversion
- Otsu's thresholding
- Morphological denoising
- Resize & normalization

All done **on-the-fly during training/testing**.

### Train Model

```bash
python train.py
```

Trains for 60 epochs with:

- Batch size: 64
- Learning rate: 0.0008 (with ReduceLROnPlateau scheduler)
- Character set: 0-9, a-z, A-Z (62 total)
- Text length: 3-7 characters (variable)
- Epochs: 60
- Uses BiLSTM + optional self-attention (enabled by default)
- CTC loss with automatic alignment
- Best model saved to `models/captcha_model_v4.pth`

### Make Predictions

```bash
python predict.py <image_path>
```

Example:

```bash
python predict.py data/train/raw/abc123_0.png --use-attention
```

Output:

```
Predicted: abc123
Ground Truth: abc123
Correct: ✓
```

### Batch Evaluation

```bash
python evaluate.py --model models/captcha_model_v4.pth --data-dir data/test/raw --use-attention
```

Options:

- `--data-dir`: Directory with raw images (default: `data/test/raw`)
- `--max-samples`: Limit number of samples to evaluate
- `--use-attention`: Enable attention mechanism in model

### Analyze Character Confusion

```bash
python analyze_confusion.py --model models/captcha_model_v4.pth --data-dir data/test/raw
```

Analyzes which characters are most often confused (e.g., 0/O, 1/I/l) and generates a detailed confusion matrix.

### Interactive Web App (Gradio)

```bash
python app.py
```

Opens at `http://localhost:7860` with:

- Image upload interface
- Real-time predictions
- Preprocessing preview
- Ground truth comparison
- Beautiful, responsive UI

## 🎯 Confusion Mitigation (0/O, 1/I/l, etc.)

The model implements several strategies to reduce confusion between visually similar characters:

### 1. Confusion-Aware Data Generation

- 40% of training samples include confusing character pairs
- Targeted generation for pairs like: 0/O, 1/I/l, 5/S, 8/B, 6/b, 2/Z
- See [generate_dataset.py](generate_dataset.py#L16-L22)

### 2. Enhanced Data Augmentation

- Stronger rotation (±8°), affine transforms, and blur
- Random noise injection (30% of samples)
- Forces model to learn robust distinctive features
- See [train.py](train.py#L280-L290)

### 3. Confusion Analysis Tool

- Character-level confusion matrix
- Identifies most problematic character pairs
- Helps monitor training effectiveness
- Run: `python analyze_confusion.py`

**Why not just remove confusing characters?**

Real-world CAPTCHAs include these characters, so the model must learn to distinguish them. Our approach teaches the model to focus on subtle differences (e.g., 0 is rounder, O is taller).

## ⚙️ Configuration

Edit these scripts to customize:

**`generate_dataset.py`**

```python
NUM_SAMPLES = 10000      # Number of images
MIN_LENGTH = 3           # Minimum text length
MAX_LENGTH = 7           # Maximum text length
# 40% samples include confusing chars (0/O, 1/I/l, etc.)
```

**`train.py`**

```python
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 0.0008
USE_LSTM = True          # Set False for CNN-only model
USE_ATTENTION = True     # Self-attention on top of BiLSTM outputs
# Stronger augmentation to distinguish 0/O, 1/I/l, etc.
```

## 📊 Performance

| Dataset    | Images | Accuracy |
| ---------- | ------ | -------- |
| Training   | 10,000 | ~90%     |
| Test (Std) | 1,000  | 83.70%   |
| Font Test  | 2,300  | 13.13%   |

**Note**: Model performs well on standard fonts but struggles with decorative/stylized fonts, highlighting the importance of font diversity in training data.

## 🧪 Checkpoints

- [models/captcha_model_v4.pth](models/captcha_model_v4.pth): Latest model with variable-length and lowercase support

## 🛠️ Technology Stack

- **PyTorch 2.0+** - Deep learning framework
- **torchvision** - Image processing
- **python-captcha** - CAPTCHA generation
- **Pillow** - Image manipulation
- **OpenCV** - Advanced image processing
- **NumPy** - Array operations

## 📝 Notes

- Images are 60×160 grayscale (auto-resized in app if needed)
- Variable-length CAPTCHAs (3-7 characters): digits (0-9) + lowercase (a-z) + uppercase (A-Z)
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
