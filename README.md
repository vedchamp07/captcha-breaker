# CAPTCHA Breaker

A deep learning project that generates and solves CAPTCHA images using PyTorch.

## 🎯 Project Overview

This project uses a Convolutional Neural Network (CNN) to recognize text in CAPTCHA images. It includes:

- CAPTCHA image generation using the `python-captcha` library
- A PyTorch-based CNN model for character recognition
- Training and prediction scripts
- Support for Kaggle GPU training

## 📁 Project Structure

```
captcha-breaker/
├── data/
│   ├── raw/              # Generated CAPTCHA images
│   └── processed/        # Preprocessed data (if needed)
├── src/
│   ├── __init__.py
│   ├── dataset.py        # PyTorch Dataset class
│   └── model.py          # CNN model architecture
├── models/               # Saved model checkpoints
├── notebooks/            # Jupyter notebooks for experiments
├── generate_dataset.py   # Generate training data
├── train.py             # Training script
├── predict.py           # Prediction script
├── requirements.txt     # Python dependencies
└── README.md
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/captcha-breaker.git
cd captcha-breaker
```

### 2. Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
pip install -r requirements.txt
```

### 3. Generate Dataset

```bash
python generate_dataset.py
```

This will create 10,000 CAPTCHA images in `data/raw/`.

### 4. Preprocess Images (Two-Stage Model)

For the two-stage model, preprocess images to grayscale and remove noise:

```bash
python preprocess.py
```

This will:

- Convert images to grayscale
- Remove background dots/noise
- Enhance contrast with CLAHE
- Denoise with fastNlMeans
- Save to `data/processed/`

### 5. Train the Model

**Two-Stage Model (Recommended):**

```bash
python train_twostage.py
```

This trains in two stages:

1. **Stage 1:** Bounding box detector (15 epochs)
2. **Stage 2:** Full model with character recognition (20 epochs)

**Original Single-Stage Model:**

```bash
python train.py
```

**On Kaggle:**

1. Upload this project to Kaggle
2. Enable GPU accelerator in Kaggle settings
3. Run preprocessing: `!python preprocess.py` (for two-stage model)
4. Run training: `!python train_twostage.py` or `!python train.py`

### 6. Make Predictions

```bash
python predict.py data/raw/ABC12_0.png
```

## 🧠 Model Architecture

### **Two-Stage Architecture** (Recommended)

The new two-stage model provides better accuracy through specialized components:

**Stage 1: Bounding Box Detector**

- Detects individual character locations with rotation
- CNN backbone (4 conv blocks) → Bbox regression head
- Outputs 5 bounding boxes: (x, y, width, height, angle)

**Stage 2: Character Recognizer**

- Recognizes normalized character patches (40×40)
- Lightweight CNN (3 conv blocks) → Classification head
- Trained on extracted character regions

**Input:** Grayscale preprocessed images (1 × 60 × 160)  
**Output:** 5 characters from [0-9, A-Z]

**Preprocessing:**

- Grayscale conversion
- Noise removal (morphological operations)
- Contrast enhancement (CLAHE)
- Denoising (fastNlMeans)

**Training/Validation Split:** 80% train, 20% validation

- Uses PyTorch's `random_split()` for reproducibility
- Recommended: Use test set from separate directory for final evaluation

### **Single-Stage Architecture** (Legacy)

Basic CNN with:

- 4 convolutional blocks with batch normalization and max pooling
- 2 fully connected layers (512→256) with batch norm and dropout
- 5 output heads (one per character position)
- Works on RGB images directly

## 📊 Results

Training metrics will appear after running `python train.py`:

- Each epoch shows training and validation accuracy
- Best model is automatically saved to `models/captcha_model.pth`

(Update after training with your actual results)

## 🔧 Configuration

Edit these variables in the scripts to customize:

**generate_dataset.py:**

- `NUM_SAMPLES`: Number of images to generate
- `CAPTCHA_LENGTH`: Length of CAPTCHA text

**train.py:**

- `BATCH_SIZE`: Batch size for training
- `EPOCHS`: Number of training epochs
- `LEARNING_RATE`: Learning rate for optimizer

## 📝 To-Do

- [ ] Add data augmentation
- [ ] Experiment with different architectures
- [ ] Add CTC loss for variable-length sequences
- [ ] Create web demo
- [ ] Add more character sets

## 🤝 Contributing

Feel free to open issues or submit pull requests!

## 📄 License

MIT License

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- python-captcha library for CAPTCHA generation
