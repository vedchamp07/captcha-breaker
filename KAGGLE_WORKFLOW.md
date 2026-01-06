# 🔐 CAPTCHA Breaker - Kaggle Workflow Guide

## 📁 Project Structure (After Cleanup)

```
captcha-breaker/
├── src/
│   ├── __init__.py
│   └── model.py              # CTC-based CAPTCHA model
├── data/
│   ├── raw/                  # Original CAPTCHA images
│   └── processed/            # Preprocessed images
├── models/
│   └── captcha_model.pth     # Trained model weights
├── notebooks/
│   └── kaggle_training.ipynb # Kaggle notebook (see below)
├── generate_dataset.py       # Generate synthetic CAPTCHAs
├── preprocess.py             # Preprocess images
├── train.py                  # Train the model
├── predict.py                # Predict single image
├── requirements.txt          # Dependencies
├── README.md                 # Project documentation
└── ARCHITECTURE_COMPARISON.md # Architecture explanation
```

---

## 🚀 Complete Kaggle Workflow

### **Phase 1: Setup on Kaggle**

#### Step 1: Create Kaggle Notebook

1. Go to https://www.kaggle.com/code
2. Click **"New Notebook"**
3. Settings → Accelerator → **GPU T4 x2** (for faster training)
4. Settings → Internet → **On** (to install packages)

#### Step 2: Upload Your Code

In Kaggle, you have 2 options:

**Option A: Upload as Kaggle Dataset** (Recommended)

1. Create a ZIP of your code:

   ```bash
   # On your local machine
   cd /Users/vedantn/captcha-breaker
   zip -r captcha-code.zip src/ generate_dataset.py preprocess.py train.py predict.py requirements.txt -x "*.pyc" "__pycache__/*"
   ```

2. Go to https://www.kaggle.com/datasets
3. Click **"New Dataset"** → Upload `captcha-code.zip`
4. Title: "CAPTCHA Breaker Code"
5. In your notebook: **Add Data** → Your Datasets → "CAPTCHA Breaker Code"

**Option B: Copy-Paste Code** (Quick & Dirty)

- Use notebook cells to create files (see notebook template below)

---

### **Phase 2: Training on Kaggle**

#### Step 3: Run Training in Notebook

Use the notebook structure below. **Key points:**

- **Notebook cells** (with `%%writefile`): For creating Python files
- **Console commands** (with `!`): For running scripts, installing packages
- **Regular cells**: For interactive code/visualization

**When to use what:**

- `!pip install` → Install packages (notebook cell)
- `!python train.py` → Run training script (notebook cell)
- File creation → Use `%%writefile` magic (notebook cell)
- Monitoring → Print statements in script show in notebook output

#### Step 4: Monitor Training

- Training progress displays directly in notebook output
- GPU usage visible in right sidebar
- Takes ~15-30 minutes for 50 epochs

---

### **Phase 3: Save & Push to GitHub**

#### Step 5: Download Trained Model from Kaggle

**Method 1: Kaggle UI**

1. In notebook, run:
   ```python
   # Cell
   from IPython.display import FileLink
   FileLink('models/captcha_model.pth')
   ```
2. Click the link to download

**Method 2: Kaggle API** (from local machine)

```bash
# Install Kaggle CLI
pip install kaggle

# Configure API token (get from kaggle.com/settings)
mkdir -p ~/.kaggle
# Download kaggle.json from Kaggle → Settings → API → Create New Token
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download notebook output
kaggle kernels output YOUR_USERNAME/YOUR_NOTEBOOK_NAME -p ./models/
```

#### Step 6: Push Everything to GitHub

**On Kaggle (using notebook cells):**

```python
# Cell 1: Configure Git
!git config --global user.email "your.email@gmail.com"
!git config --global user.name "Your Name"
```

```python
# Cell 2: Clone your repo (if not already)
!git clone https://github.com/YOUR_USERNAME/captcha-breaker.git
%cd captcha-breaker
```

```python
# Cell 3: Copy trained model
!mkdir -p models
!cp ../working/models/captcha_model.pth models/
```

```python
# Cell 4: Commit and push
!git add models/captcha_model.pth
!git commit -m "Add trained model from Kaggle"
!git push origin main
```

**Authentication on Kaggle:**
GitHub requires a Personal Access Token (PAT) for push:

```python
# Cell: Setup GitHub token
import os
from kaggle_secrets import UserSecretsClient

# Add your GitHub token to Kaggle Secrets first:
# Notebook → Add-ons → Secrets → + Add a new secret
# Label: GITHUB_TOKEN
# Value: your_github_personal_access_token

user_secrets = UserSecretsClient()
github_token = user_secrets.get_secret("GITHUB_TOKEN")

# Configure git to use token
!git remote set-url origin https://{github_token}@github.com/YOUR_USERNAME/captcha-breaker.git
!git push origin main
```

**How to create GitHub Personal Access Token:**

1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token → Select scopes: `repo` (all)
3. Copy token → Add to Kaggle Secrets

---

### **Phase 4: Pull to Local Machine**

```bash
# On your local machine
cd /Users/vedantn/captcha-breaker

# Pull the trained model
git pull origin main

# Now you have the trained model locally!
ls -lh models/captcha_model.pth
```

---

## 📓 Kaggle Notebook Template

Save this as `notebooks/kaggle_training.ipynb` or create it directly in Kaggle:

### **Cell 1: Setup Environment** (Code Cell)

```python
# Check GPU
!nvidia-smi

# Install dependencies
!pip install -q torch torchvision captcha opencv-python
```

### **Cell 2: Create Project Structure** (Code Cell)

```python
import os
os.makedirs('src', exist_ok=True)
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)
```

### **Cell 3: Copy Code Files** (Multiple cells with %%writefile)

**If you uploaded code as dataset:**

```python
# Unzip code dataset
!unzip -q /kaggle/input/captcha-breaker-code/captcha-code.zip -d .
!ls -la
```

**If copying manually:**

```python
%%writefile src/__init__.py
# Empty file
```

```python
%%writefile src/model.py
# Paste entire content of src/model.py here
```

```python
%%writefile train.py
# Paste entire content of train.py here
```

### **Cell 4: Generate Dataset** (Code Cell)

```python
# Generate synthetic CAPTCHAs
!python generate_dataset.py
!ls -lh data/raw/ | head
```

### **Cell 5: Preprocess Data** (Code Cell)

```python
# Preprocess images
!python preprocess.py
!ls -lh data/processed/ | head
```

### **Cell 6: Train Model** (Code Cell)

```python
# Train the model (this takes 15-30 minutes)
!python train.py
```

### **Cell 7: Check Results** (Code Cell)

```python
import torch

# Load model and check
model_path = 'models/captcha_model.pth'
checkpoint = torch.load(model_path)
print(f"Model saved successfully: {model_path}")
print(f"File size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
```

### **Cell 8: Test Prediction** (Code Cell)

```python
# Test on a sample image
!python predict.py data/processed/ABC12_0.png
```

### **Cell 9: Download Model** (Code Cell)

```python
from IPython.display import FileLink
FileLink('models/captcha_model.pth')
```

---

## 🎯 Quick Command Reference

### **In Kaggle Notebook:**

```python
# Install packages
!pip install package-name

# Run Python scripts
!python script.py

# Create files
%%writefile filename.py
# content here

# Check GPU
!nvidia-smi

# List files
!ls -lh

# Git operations
!git add .
!git commit -m "message"
!git push origin main
```

### **In Kaggle Console (Terminal):**

Click **Console** button in Kaggle notebook for terminal access:

```bash
# Same as notebook, but without ! prefix
python train.py
ls -la
git status
```

### **On Local Machine:**

```bash
# Pull updates
git pull origin main

# Test model
python predict.py data/processed/test.png
```

---

## 📊 Expected Results

After training on Kaggle:

- **Training time**: 15-30 minutes (50 epochs on GPU)
- **Expected accuracy**: 50-90% (much better than 6.25% with broken bbox approach!)
- **Model size**: ~5-20 MB
- **Validation loss**: Should decrease steadily

---

## ⚠️ Troubleshooting

### Out of Memory on Kaggle

```python
# In train.py, reduce batch size:
BATCH_SIZE = 32  # or even 16
```

### Git Push Failed

```python
# Make sure token is in Kaggle Secrets
# Or push manually after downloading model
```

### Import Errors

```python
# Make sure src/__init__.py exists
# Check file paths are correct
```

---

## 🎓 Summary: Console vs Notebook

| Task                | Where to Run    | How to Run               |
| ------------------- | --------------- | ------------------------ |
| Install packages    | Notebook Cell   | `!pip install torch`     |
| Create Python files | Notebook Cell   | `%%writefile train.py`   |
| Run training script | Notebook Cell   | `!python train.py`       |
| Monitor progress    | Notebook Output | Automatic display        |
| Interactive code    | Notebook Cell   | Regular Python code      |
| Git commands        | Notebook Cell   | `!git add .`             |
| Terminal work       | Kaggle Console  | `python train.py` (no !) |
| Download files      | Notebook Cell   | `FileLink('model.pth')`  |

**General Rule**: Everything can be done in notebook cells with `!` prefix. Use Kaggle Console (terminal) only if you prefer terminal workflow.
