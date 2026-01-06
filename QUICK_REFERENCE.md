# 📋 Quick Reference: Kaggle Console vs Notebook Cells

## When to Use Console vs Notebook Cells

### ✅ Use **Notebook Cells** (Recommended)

**For everything!** All commands can run in notebook cells with `!` prefix:

```python
# Install packages
!pip install torch torchvision

# Run scripts
!python train.py

# Create files
%%writefile script.py
# code here

# Git commands
!git add .
!git commit -m "message"
!git push

# File operations
!ls -la
!mkdir data
!cp file1.py file2.py
```

**Advantages:**

- Output saved in notebook
- Can rerun cells
- Easy to share
- All history preserved

### ⚙️ Use **Console/Terminal** (Optional)

Click "Console" button in Kaggle for terminal access. Use when:

- You prefer terminal workflow
- Need interactive shell
- Debugging issues
- Long-running commands

**In console, remove `!` prefix:**

```bash
# Same commands, no ! needed
python train.py
ls -la
git status
```

---

## 🎯 Command Reference

| Task            | Notebook Cell                          | Console                        |
| --------------- | -------------------------------------- | ------------------------------ |
| Install package | `!pip install torch`                   | `pip install torch`            |
| Run script      | `!python train.py`                     | `python train.py`              |
| Create file     | `%%writefile file.py`                  | `nano file.py` or `vi file.py` |
| List files      | `!ls -la`                              | `ls -la`                       |
| Git clone       | `!git clone URL`                       | `git clone URL`                |
| Git push        | `!git push`                            | `git push`                     |
| Check GPU       | `!nvidia-smi`                          | `nvidia-smi`                   |
| Change dir      | `%cd /path` or `import os; os.chdir()` | `cd /path`                     |
| Environment var | `import os; os.environ['VAR']='val'`   | `export VAR=val`               |

---

## 📝 Step-by-Step Kaggle Workflow

### **Phase 1: Setup** (Notebook Cells)

```python
# Cell 1: Check GPU
!nvidia-smi

# Cell 2: Install
!pip install -q torch torchvision captcha opencv-python

# Cell 3: Create directories
import os
os.makedirs('src', exist_ok=True)
os.makedirs('data/raw', exist_ok=True)
os.makedirs('models', exist_ok=True)
```

### **Phase 2: Code Upload** (Notebook Cells)

**Option A: From dataset**

```python
# Cell: Unzip uploaded code
!unzip -q /kaggle/input/your-dataset/code.zip -d .
```

**Option B: Create files**

```python
# Cell: Create each file
%%writefile src/model.py
# paste code here
```

### **Phase 3: Training** (Notebook Cells)

```python
# Cell 1: Generate data
!python generate_dataset.py

# Cell 2: Preprocess
!python preprocess.py

# Cell 3: Train (takes 15-30 min)
!python train.py
```

### **Phase 4: Save Model** (Notebook Cells)

```python
# Cell 1: Check model
import os
model_size = os.path.getsize('models/captcha_model.pth') / (1024*1024)
print(f"Model size: {model_size:.2f} MB")

# Cell 2: Create download link
from IPython.display import FileLink
FileLink('models/captcha_model.pth')
```

### **Phase 5: Push to GitHub** (Notebook Cells)

```python
# Cell 1: Configure git
!git config --global user.email "you@example.com"
!git config --global user.name "Your Name"

# Cell 2: Setup authentication
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GITHUB_TOKEN")

# Cell 3: Clone/push
!git clone https://{token}@github.com/user/repo.git
%cd repo
!cp ../models/captcha_model.pth models/
!git add models/captcha_model.pth
!git commit -m "Add trained model"
!git push origin main
```

---

## 🔐 GitHub Token Setup

### 1. Create Personal Access Token

1. GitHub.com → Settings → Developer settings
2. Personal access tokens → Tokens (classic)
3. Generate new token
4. Select scopes: **repo** (all checkboxes)
5. Generate → **Copy token**

### 2. Add to Kaggle Secrets

1. In Kaggle notebook → **Add-ons** → **Secrets**
2. Click **+ Add a new secret**
3. Label: `GITHUB_TOKEN`
4. Value: paste your token
5. Click **Add**

### 3. Use in Notebook

```python
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
!git remote set-url origin https://{token}@github.com/user/repo.git
```

---

## 💡 Pro Tips

### Notebook Cell Magic Commands

```python
# Write file
%%writefile file.py
code here

# Time execution
%%time
!python train.py

# Capture output
output = !ls -la

# Change directory (persistent)
%cd /kaggle/working
```

### File Operations

```python
# Check if file exists
import os
if os.path.exists('models/model.pth'):
    print("Model found!")

# List files
import glob
files = glob.glob('data/raw/*.png')
print(f"Found {len(files)} images")

# Read file size
size = os.path.getsize('file.txt')
print(f"Size: {size/1024:.2f} KB")
```

### Monitor Training

```python
# In train.py, use tqdm for progress bars
from tqdm import tqdm

for epoch in tqdm(range(epochs)):
    # training code
    pass
```

---

## 🚨 Common Issues

### Import Error: Module not found

```python
# Make sure src/__init__.py exists
!touch src/__init__.py

# Or add to path
import sys
sys.path.append('/kaggle/working')
```

### Git Push Failed

```python
# Check token is set
from kaggle_secrets import UserSecretsClient
try:
    token = UserSecretsClient().get_secret("GITHUB_TOKEN")
    print("Token found!")
except:
    print("Add GITHUB_TOKEN to Kaggle Secrets")
```

### Out of Memory

```python
# Reduce batch size in train.py
BATCH_SIZE = 32  # or 16
```

### File Not Found

```python
# Check current directory
!pwd
!ls -la

# Use absolute paths
model_path = '/kaggle/working/models/captcha_model.pth'
```

---

## ✅ Final Checklist

Before running on Kaggle:

- [ ] GPU enabled in notebook settings
- [ ] Internet enabled for pip install
- [ ] All code files uploaded or created
- [ ] GitHub token added to Kaggle Secrets
- [ ] Git email/name configured
- [ ] Directory structure created

During training:

- [ ] Monitor GPU usage (sidebar)
- [ ] Check loss decreasing
- [ ] Validate accuracy improving
- [ ] Save checkpoints

After training:

- [ ] Download model via FileLink
- [ ] Push to GitHub
- [ ] Pull to local machine
- [ ] Test predictions locally

---

**Remember**: Everything in notebook cells is saved and can be rerun. Use cells for reproducibility!
