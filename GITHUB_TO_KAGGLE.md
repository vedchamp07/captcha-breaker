# 🔗 GitHub to Kaggle Workflow

## ✅ Step-by-Step: Import from GitHub and Run on Kaggle

### **Step 1: Your Code is on GitHub** ✓

Your repository: `https://github.com/vedchamp07/captcha-breaker`

**Bug Fixed**: Device mismatch error resolved (predictions now created on same device as input)

---

### **Step 2: Create Kaggle Notebook**

1. Go to https://www.kaggle.com/code
2. Click **"New Notebook"**
3. **Settings** (gear icon):
   - **Accelerator**: GPU T4 x2 (or P100)
   - **Internet**: **ON** (required!)
   - **Persistence**: Files only

---

### **Step 3: Clone from GitHub in Kaggle**

#### **Cell 1: Check GPU**

```python
!nvidia-smi
```

#### **Cell 2: Clone Private Repository with GitHub Token**

Since your repo is private, use your GitHub token stored in Kaggle Secrets:

```python
# Get GitHub token from Kaggle Secrets
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
github_token = user_secrets.get_secret("GITHUB_TOKEN")

# Clone your private GitHub repo using token authentication
!git clone https://{github_token}@github.com/vedchamp07/captcha-breaker.git
%cd captcha-breaker

# Verify files
!ls -la
```

**What's happening:**

- `UserSecretsClient()` retrieves your `GITHUB_TOKEN` from Kaggle Secrets
- The token is inserted into the URL for authentication
- Works for both public and private repos!

#### **Cell 3: Install Dependencies**

```python
!pip install -q torch torchvision captcha opencv-python tqdm
```

---

### **Step 4: Create Directories**

```python
import os

# Create data and model directories
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("✓ Directories created")
```

---

### **Step 5: Generate Dataset**

```python
# Generate 10,000 CAPTCHA images
!python generate_dataset.py

# Check generated files
!echo "Generated images:"
!ls data/raw/ | head -5
!echo "Total: $(ls data/raw/ | wc -l) images"
```

---

### **Step 6: Preprocess Images**

```python
# Convert to grayscale and remove noise
!python preprocess.py

# Check preprocessed files
!echo "Preprocessed images:"
!ls data/processed/ | head -5
!echo "Total: $(ls data/processed/ | wc -l) images"
```

---

### **Step 7: Train Model** 🚀

```python
# This takes 15-30 minutes on GPU
!python train.py
```

**Expected output:**

- Training and validation progress bars
- Loss decreasing
- Accuracy increasing to 50-90%
- Best model saved to `models/captcha_model.pth`

---

### **Step 8: Check Results**

```python
import torch
import os

# Verify model was saved
model_path = 'models/captcha_model.pth'

if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"✓ Model saved successfully")
    print(f"  Path: {model_path}")
    print(f"  Size: {size_mb:.2f} MB")

    # Load and check
    checkpoint = torch.load(model_path, map_location='cpu')
    print(f"  Model loaded successfully!")
else:
    print("✗ Model file not found!")
```

---

### **Step 9: Test Predictions**

```python
# Test on a few samples
import glob

sample_images = glob.glob('data/processed/*.png')[:5]

print("Testing predictions:\n")
for img_path in sample_images:
    print(f"Image: {os.path.basename(img_path)}")
    !python predict.py {img_path}
    print()
```

---

### **Step 10: Visualize Results** (Optional)

```python
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
import string
from src.model import CTCCaptchaModel

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
characters = string.digits + string.ascii_uppercase

# Load model
model = CTCCaptchaModel(num_classes=len(characters))
model.load_state_dict(torch.load('models/captcha_model.pth', map_location=device))
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((60, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Visualize 6 samples
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

sample_images = glob.glob('data/processed/*.png')[:6]

for idx, img_path in enumerate(sample_images):
    # Load and predict
    img = Image.open(img_path).convert('L')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_indices = model.predict(img_tensor)[0]
        pred_text = ''.join([characters[i] for i in pred_indices if i < len(characters)])

    # Ground truth
    gt_text = os.path.basename(img_path).split('_')[0]

    # Plot
    axes[idx].imshow(img, cmap='gray')
    is_correct = pred_text == gt_text
    axes[idx].set_title(
        f'GT: {gt_text}\nPred: {pred_text}',
        fontsize=12,
        color='green' if is_correct else 'red',
        fontweight='bold'
    )
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Visualization saved to predictions.png")
```

---

### **Step 11: Download Trained Model**

```python
from IPython.display import FileLink

# Create download link
print("Click to download your trained model:")
FileLink('models/captcha_model.pth')
```

---

### **Step 12: Push Model to GitHub** (Optional)

#### **Setup GitHub Token:**

1. GitHub → Settings → Developer settings → Personal access tokens
2. Generate new token → Select **`repo`** scope
3. Copy token
4. Kaggle notebook → **Add-ons** → **Secrets** → Add secret:
   - Label: `GITHUB_TOKEN`
   - Value: paste your token

#### **In Kaggle Cell:**

```python
# Configure Git
!git config --global user.email "your.email@example.com"
!git config --global user.name "Your Name"

# Setup authentication
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GITHUB_TOKEN")

# Set remote with token
!git remote set-url origin https://{token}@github.com/vedchamp07/captcha-breaker.git

# Commit and push model
!git add models/captcha_model.pth
!git commit -m "Add trained CAPTCHA model from Kaggle"
!git push origin main

print("✓ Model pushed to GitHub!")
```

---

### **Step 13: Pull to Local Machine**

```bash
# On your local machine
cd /Users/vedantn/captcha-breaker
git pull origin main

# Test the model
python predict.py data/processed/test.png
```

---

## 🎯 Quick Command Summary

| Step          | Command                                                   | Where          |
| ------------- | --------------------------------------------------------- | -------------- |
| Clone repo    | See **Cell 2** (uses GitHub token from Secrets)           | Kaggle cell    |
| Install deps  | `!pip install -q torch torchvision captcha opencv-python` | Kaggle cell    |
| Generate data | `!python generate_dataset.py`                             | Kaggle cell    |
| Preprocess    | `!python preprocess.py`                                   | Kaggle cell    |
| Train         | `!python train.py`                                        | Kaggle cell    |
| Test          | `!python predict.py data/processed/ABC12_0.png`           | Kaggle cell    |
| Download      | `FileLink('models/captcha_model.pth')`                    | Kaggle cell    |
| Pull locally  | `git pull origin main`                                    | Local terminal |

---

## ⚡ Why GitHub → Kaggle is Better

**Advantages:**

- ✅ Always have latest code (just git pull)
- ✅ Easy to update and re-run
- ✅ No need to zip/upload manually
- ✅ Version control built-in
- ✅ Can push trained model back to GitHub
- ✅ Team collaboration ready

**vs Uploading as Dataset:**

- ❌ Must re-upload every time you change code
- ❌ Version management harder
- ❌ More steps to update

---

## 🐛 Bug Fixed!

**Problem**: `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`

**Root cause**: In `src/model.py`, the `predict()` method was creating tensors on CPU:

```python
return torch.tensor(padded, dtype=torch.long)  # ❌ CPU
```

**Fix**: Now creates tensors on same device as input:

```python
return torch.tensor(padded, dtype=torch.long, device=x.device)  # ✅ GPU/CPU
```

This fix is already in your GitHub repo! 🎉

---

## 🎓 Expected Training Results

After running `!python train.py` on Kaggle:

- **Duration**: 15-30 minutes (50 epochs on GPU)
- **Final accuracy**: 50-90% validation
- **Model size**: ~5-20 MB
- **Much better than**: 6.25% from broken bbox approach

---

## ✅ Complete Notebook Template (for Private Repo)

Copy this entire sequence into a new Kaggle notebook:

```python
# Cell 1: GPU Check
!nvidia-smi

# Cell 2: Clone Private Repo with Token
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
github_token = user_secrets.get_secret("GITHUB_TOKEN")
!git clone https://{github_token}@github.com/vedchamp07/captcha-breaker.git
%cd captcha-breaker

# Cell 3: Install
!pip install -q torch torchvision captcha opencv-python tqdm

# Cell 4: Setup
import os
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Cell 5: Generate dataset
!python generate_dataset.py

# Cell 6: Preprocess
!python preprocess.py

# Cell 7: Train (15-30 min)
!python train.py

# Cell 8: Check model
import torch
size = os.path.getsize('models/captcha_model.pth') / (1024*1024)
print(f"✓ Model: {size:.2f} MB")

# Cell 9: Test
!python predict.py data/processed/ABC12_0.png

# Cell 10: Download
from IPython.display import FileLink
FileLink('models/captcha_model.pth')
```

That's it! 🚀
