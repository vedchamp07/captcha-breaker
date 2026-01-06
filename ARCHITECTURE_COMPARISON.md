# CAPTCHA Recognition - Approach Comparison

## ❌ Problem with Two-Stage (Bounding Box) Approach

**Critical Issue**: Ground truth bounding boxes are **uniform approximations** but real CAPTCHAs have:

- Variable character spacing
- Different sizes and rotations
- Overlapping characters
- Unpredictable positions after preprocessing

**Result**: Model learns incorrect bounding boxes → poor performance

---

## ✅ Solution: Better Architectures (No Bounding Boxes)

### **Option 1: CTC-Based Model** ⭐ **RECOMMENDED**

**What is CTC?**

- Connectionist Temporal Classification
- Industry standard for OCR/CAPTCHA recognition
- **No bounding boxes needed!**

**How it works:**

1. CNN extracts visual features
2. Features reshaped as sequence (width → time steps)
3. LSTM processes sequence
4. CTC loss automatically aligns predictions with targets
5. Model learns character positions without explicit supervision

**Advantages:**

- ✅ No manual labeling needed
- ✅ Handles variable spacing automatically
- ✅ Proven approach (used in Google Tesseract, etc.)
- ✅ Works with overlapping/distorted characters

**Files created:**

- `src/model_ctc.py` - CTC model (LSTM version + simple version)
- `train_ctc.py` - Training script
- `predict_ctc.py` - Prediction script

**Usage:**

```bash
# Train
python train_ctc.py

# Predict
python predict_ctc.py data/raw/ABC12_0.png
```

---

### **Option 2: Keep Original Simple CNN** ✅ **Simplest**

**Your original approach** in `train.py` was actually fine!

- Direct prediction: CNN → 5 character classifications
- No complexity of bounding boxes or sequences
- Just needed better preprocessing + normalization (which we fixed)

**When to use:** If CTC seems complex, stick with the original `train.py` approach

---

### **Option 3: Attention-Based Model** 🔬 **Advanced**

**How it works:**

- Encoder (CNN) extracts features
- Decoder (RNN) generates characters one by one
- Attention mechanism learns where to "look" for each character
- No explicit bounding boxes, model learns attention regions

**When to use:** Research/experimentation, more complex but potentially best accuracy

---

### **Option 4: YOLO for Pseudo-Labeling** ⚠️ **Not Recommended**

**Your suggestion to use YOLO:**

- YOLO is for object detection (trained on real-world objects)
- Would need fine-tuning on CAPTCHA characters
- More complex than CTC approach
- Still requires some labeled data for fine-tuning

**Verdict:** Possible but unnecessary - CTC is better for this task

---

## 🎯 Recommendation

**Use the CTC approach** (Option 1):

1. **Easy to implement** - I've created all the code for you
2. **No labeling required** - just image filename labels (which you already have)
3. **Industry standard** - proven to work for sequence recognition
4. **Better than two-stage** - no incorrect bbox assumptions

**Quick comparison:**

| Approach              | Bbox Needed?     | Labeling Effort | Complexity | Expected Accuracy |
| --------------------- | ---------------- | --------------- | ---------- | ----------------- |
| Two-stage (current)   | ✅ Yes (broken!) | High            | High       | ❌ Poor           |
| **CTC (recommended)** | ❌ No            | None            | Medium     | ✅ Good           |
| Simple CNN (original) | ❌ No            | None            | Low        | ✅ OK             |
| Attention             | ❌ No            | None            | High       | ✅ Best           |
| YOLO                  | ✅ Yes           | Medium          | Very High  | ❓ Unknown        |

---

## 🚀 Next Steps

### Switch to CTC approach:

```bash
# 1. Train CTC model (uses existing data)
python train_ctc.py

# 2. Test on sample
python predict_ctc.py data/processed/ABC12_0.png

# 3. Compare with old approach
# Old: 6.25% accuracy (broken bboxes)
# Expected: 50-90% accuracy (proper CTC)
```

### Or keep it simple:

```bash
# Just retrain the original model with fixed preprocessing
python train.py
```

The two-stage architecture was a good idea in theory, but CTC is the proven solution for this exact problem! 🎯
