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

### 4. Train the Model

**Local (Mac):**
```bash
python train.py
```

**On Kaggle:**
1. Upload this project to Kaggle
2. Enable GPU accelerator in Kaggle settings
3. Run the training notebook or script

### 5. Make Predictions

```bash
python predict.py data/raw/ABC12_0.png
```

## 🧠 Model Architecture

The model uses a CNN with:
- 4 convolutional blocks with max pooling
- 2 fully connected layers with dropout
- 5 output heads (one per character position)

Input: RGB images (3 × 60 × 160)
Output: 5 characters from [0-9, A-Z]

## 📊 Results

- Training accuracy: ~XX%
- Validation accuracy: ~XX%

(Update after training)

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