"""
Training script for CTC-based CAPTCHA model.
No bounding boxes needed!
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision import transforms
from PIL import Image
import string
from pathlib import Path
from tqdm import tqdm
import numpy as np

from src.model import CTCCaptchaModel, CTCCaptchaModelSimple


# Configuration
DATA_DIR = Path("data/processed")  # or data/raw
MODEL_SAVE_PATH = Path("models/captcha_model.pth")
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 0.0008
TRAIN_SPLIT = 0.8
USE_LSTM = True  # Set False for simpler/faster model
USE_ATTENTION = True  # Enable self-attention on top of LSTM

# Character set
CHARACTERS = string.digits + string.ascii_uppercase
NUM_CLASSES = len(CHARACTERS)


class SimpleCaptchaDataset(Dataset):
    """Simple dataset - just images and labels, no bboxes!"""
    
    def __init__(self, data_dir, characters, transform=None):
        self.data_dir = Path(data_dir)
        self.characters = characters
        self.transform = transform
        
        self.char_to_idx = {char: idx for idx, char in enumerate(characters)}
        self.image_files = sorted(list(self.data_dir.glob("*.png")))
        
        print(f"Found {len(self.image_files)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # Extract label from filename
        label_text = img_path.stem.split('_')[0]
        
        # Load image
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        # Convert label to indices
        label = [self.char_to_idx[c] for c in label_text]
        
        return image, torch.tensor(label, dtype=torch.long), label_text


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels, label_texts in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        log_probs = model(images)  # (seq_len, batch, num_classes+1)
        
        # Prepare for CTC loss
        batch_size = images.size(0)
        input_lengths = torch.full((batch_size,), log_probs.size(0), dtype=torch.long)
        target_lengths = torch.full((batch_size,), labels.size(1), dtype=torch.long)
        
        # CTC loss
        loss = criterion(log_probs, labels, input_lengths, target_lengths)
        
        # Backward pass
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        predictions = model.predict(images)
        model.train()  # ← IMPORTANT: restore training mode after predict()
        correct += (predictions == labels).all(dim=1).sum().item()
        total += batch_size
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    return total_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, labels, label_texts in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            log_probs = model(images)
            
            # CTC loss
            batch_size = images.size(0)
            input_lengths = torch.full((batch_size,), log_probs.size(0), dtype=torch.long)
            target_lengths = torch.full((batch_size,), labels.size(1), dtype=torch.long)
            
            loss = criterion(log_probs, labels, input_lengths, target_lengths)
            total_loss += loss.item()
            
            # Calculate accuracy (decode without using predict() to avoid mode changes)
            _, preds = log_probs.max(2)  # (seq_len, batch)
            preds = preds.transpose(0, 1)  # (batch, seq_len)
            
            # Simple greedy decode
            decoded = []
            for pred_seq in preds:
                decoded_seq = []
                prev_char = None
                for char_idx in pred_seq:
                    char_idx = char_idx.item()
                    if char_idx == model.blank_idx:
                        prev_char = None
                        continue
                    if char_idx != prev_char:
                        decoded_seq.append(char_idx)
                        prev_char = char_idx
                decoded.append(decoded_seq)
            
            # Pad to length 5
            predictions = []
            for seq in decoded:
                if len(seq) < 5:
                    seq = seq + [0] * (5 - len(seq))
                else:
                    seq = seq[:5]
                predictions.append(seq)
            predictions = torch.tensor(predictions, dtype=torch.long, device=device)
            
            correct += (predictions == labels).all(dim=1).sum().item()
            total += batch_size
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })
    
    return total_loss / len(dataloader), correct / total


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check data directory
    if not DATA_DIR.exists() or len(list(DATA_DIR.glob("*.png"))) == 0:
        data_dir = Path("data/raw")
        print(f"Using raw data from {data_dir}")
    else:
        data_dir = DATA_DIR
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((60, 160)),
        transforms.RandomRotation(5, fill=0),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=3, fill=0),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.4)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((60, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Dataset
    base_dataset = SimpleCaptchaDataset(data_dir, CHARACTERS, transform=None)
    # Split indices deterministically
    total_len = len(base_dataset)
    train_size = int(TRAIN_SPLIT * total_len)
    indices = torch.randperm(total_len).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create train/val datasets with different transforms
    train_dataset = Subset(SimpleCaptchaDataset(data_dir, CHARACTERS, transform=train_transform), train_indices)
    val_dataset = Subset(SimpleCaptchaDataset(data_dir, CHARACTERS, transform=val_transform), val_indices)
    
    # DataLoaders
    num_workers = 2 if torch.cuda.is_available() else 0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=num_workers)
    
    # Model
    if USE_LSTM:
        print("Using CTC model with LSTM" + (" + Attention" if USE_ATTENTION else ""))
        model = CTCCaptchaModel(num_classes=NUM_CLASSES, hidden_size=256, num_lstm_layers=2, use_attention=USE_ATTENTION)
    else:
        print("Using simple CTC model (no LSTM)")
        model = CTCCaptchaModelSimple(num_classes=NUM_CLASSES)
    
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CTCLoss(blank=NUM_CLASSES, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=3)
    
    # Training loop
    best_val_acc = 0.0
    
    print(f"\nStarting training for {EPOCHS} epochs...")
    print("=" * 80)
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 80)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            MODEL_SAVE_PATH.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ✓ New best model saved! Val Acc: {best_val_acc*100:.2f}%")
    
    print("\n" + "=" * 80)
    print(f"Training complete! Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
