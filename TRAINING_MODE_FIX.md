# 🔧 Training Mode Bug Fix

## The Error

```
RuntimeError: cudnn RNN backward can only be called in training mode
```

## Root Cause

In the `train_epoch()` function, we were calling `model.predict(images)` to calculate accuracy during training. The problem:

1. `model.predict()` sets the model to **eval mode** with `self.eval()`
2. After accuracy calculation, the loop continues and tries to do `loss.backward()`
3. LSTM with cudnn backend requires **training mode** for backward pass
4. Error! ❌

## The Fix

### In `train_epoch()` (line ~100):

```python
# BEFORE ❌
predictions = model.predict(images)
correct += (predictions == labels).all(dim=1).sum().item()

# AFTER ✅
predictions = model.predict(images)
model.train()  # Restore training mode after predict()
correct += (predictions == labels).all(dim=1).sum().item()
```

Just add `model.train()` after calling `predict()` to put the model back in training mode!

### In `validate()` (line ~135):

Instead of using `model.predict()` (which manages eval/train modes), we decode the predictions **inline** while the model is already in eval mode:

```python
# BEFORE ❌
predictions = model.predict(images)
correct += (predictions == labels).all(dim=1).sum().item()

# AFTER ✅
_, preds = log_probs.max(2)  # Greedy decode
# ... manual decoding logic ...
predictions = torch.tensor(predictions, dtype=torch.long, device=device)
correct += (predictions == labels).all(dim=1).sum().item()
```

This avoids calling `predict()` which tries to set eval mode again (already in eval).

## Why This Works

- **Training**: After `backward()`, the loop needs training mode for the next iteration
- **Validation**: Already in eval mode with `with torch.no_grad()`, so decode inline

## Status

✅ Fixed in [train.py](train.py)
