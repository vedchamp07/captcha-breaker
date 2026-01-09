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
import random
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2

from src.model import CTCCaptchaModel, CTCCaptchaModelSimple


# Configuration
DATA_DIR = Path("data/train/raw")  # Use raw images - preprocessing on-the-fly
MODEL_SAVE_PATH = Path("models/captcha_model_v3.pth")
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 0.0008
TRAIN_SPLIT = 0.8
USE_LSTM = True  # Set False for simpler/faster model
USE_ATTENTION = True  # Enable self-attention on top of LSTM

# Character set (digits + lowercase + uppercase letters)
CHARACTERS = string.digits + string.ascii_lowercase + string.ascii_uppercase
NUM_CLASSES = len(CHARACTERS)

# Confusing character pairs for confusion penalty
CONFUSING_PAIRS = [
    ('0', 'O'),  # Zero and uppercase O
    ('1', 'I', 'l'),  # One, uppercase I, lowercase L
    ('5', 'S'),  # Five and uppercase S
    ('8', 'B'),  # Eight and uppercase B
    ('6', 'b'),  # Six and lowercase b
    ('2', 'Z'),  # Two and uppercase Z
]


def build_confusion_matrix(characters, confusing_pairs):
    """Build a confusion penalty matrix for similar characters.
    
    Returns a matrix where element [i][j] is the penalty for confusing char i with char j.
    """
    char_to_idx = {char: idx for idx, char in enumerate(characters)}
    n = len(characters)
    confusion_matrix = torch.zeros((n, n))
    
    # Add penalties for confusing character pairs
    for group in confusing_pairs:
        indices = [char_to_idx[c] for c in group if c in char_to_idx]
        # Add penalty between all pairs in this group
        for i in indices:
            for j in indices:
                if i != j:
                    confusion_matrix[i][j] = 2.0  # 2x penalty for confusing similar chars
    
    return confusion_matrix


def preprocess_image(image):
    """
    Preprocess image: grayscale, thresholding, morphological operations.
    
    Args:
        image: PIL Image
    
    Returns:
        Preprocessed PIL Image
    """
    # Convert to numpy array
    img_array = np.array(image.convert('L'))
    
    # Apply Otsu's thresholding
    _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological closing to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Convert back to PIL Image
    return Image.fromarray(processed)


class SimpleCaptchaDataset(Dataset):
    """Simple dataset - just images and labels, no bboxes!
    
    Preprocessing is done on-the-fly during training for efficiency.
    """
    
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
        
        # Preprocess image on-the-fly
        image = preprocess_image(image)
        
        if self.transform:
            image = self.transform(image)
        
        # Convert label to indices
        label = [self.char_to_idx[c] for c in label_text]
        
        return image, torch.tensor(label, dtype=torch.long), label_text


def collate_fn(batch):
    """Custom collate function to handle variable-length labels.
    
    Pads labels to the maximum length in the batch.
    """
    images, labels, label_texts = zip(*batch)
    
    # Stack images (all same size)
    images = torch.stack(images, 0)
    
    # Pad labels to max length in batch
    max_label_len = max(len(label) for label in labels)
    padded_labels = []
    for label in labels:
        # Pad with zeros (which we'll ignore in loss calculation)
        padding = torch.zeros(max_label_len - len(label), dtype=torch.long)
        padded_label = torch.cat([label, padding])
        padded_labels.append(padded_label)
    
    labels = torch.stack(padded_labels, 0)
    
    return images, labels, label_texts


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
        
        # Prepare for CTC loss - supports variable length targets!
        batch_size = images.size(0)
        input_lengths = torch.full((batch_size,), log_probs.size(0), dtype=torch.long)
        
        # Calculate actual target lengths for each sample (variable length support)
        target_lengths = []
        for label in labels:
            # Count non-padding tokens (padding is 0 from dataset)
            actual_length = (label != 0).sum().item()
            target_lengths.append(actual_length)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        
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
            
            # CTC loss with variable target lengths
            batch_size = images.size(0)
            input_lengths = torch.full((batch_size,), log_probs.size(0), dtype=torch.long)
            
            # Calculate actual target lengths
            target_lengths = []
            for label in labels:
                actual_length = (label != 0).sum().item()
                target_lengths.append(actual_length)
            target_lengths = torch.tensor(target_lengths, dtype=torch.long)
            
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
        data_dir = Path("data/train/raw")
        print(f"Using raw data from {data_dir}")
    else:
        data_dir = DATA_DIR
    
    # Transforms - stronger augmentation to learn distinctive features
    train_transform = transforms.Compose([
        transforms.Resize((60, 160)),
        transforms.RandomRotation(8, fill=0),  # Increased rotation
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1), shear=5, fill=0),  # Stronger affine
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),  # Slightly more blur
        transforms.ColorJitter(brightness=0.15, contrast=0.15),  # More brightness/contrast variation
        transforms.ToTensor(),
        # Random noise to force learning robust features
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.02 if random.random() < 0.3 else x),
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
                             shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    
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
