"""
Training script for two-stage CAPTCHA breaker.
Trains bounding box detector and character recognizer separately.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import string
from pathlib import Path
from tqdm import tqdm

from src.dataset_grayscale import CaptchaDatasetGrayscale
from src.model_twostage import TwoStageCaptchaModel

# Configuration
DATA_DIR = Path("data/processed")  # Use preprocessed images
MODEL_SAVE_PATH = Path("models/captcha_model_twostage.pth")
BATCH_SIZE = 32
EPOCHS_STAGE1 = 15  # Epochs for bbox detector
EPOCHS_STAGE2 = 20  # Epochs for full model
LEARNING_RATE_STAGE1 = 0.001
LEARNING_RATE_STAGE2 = 0.0005
TRAIN_SPLIT = 0.8
CAPTCHA_LENGTH = 5

# Character set
CHARACTERS = string.digits + string.ascii_uppercase
NUM_CLASSES = len(CHARACTERS)


def bbox_loss_fn(pred_bboxes, target_bboxes):
    """
    Loss function for bounding box prediction.
    Smooth L1 loss for bbox parameters.
    """
    # pred_bboxes: (batch_size, num_chars, 5)
    # target_bboxes: (batch_size, num_chars, 5)
    loss = nn.SmoothL1Loss()(pred_bboxes, target_bboxes)
    return loss


def train_stage1_epoch(model, dataloader, optimizer, device):
    """Train Stage 1 (bounding box detector) for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc="Stage 1: Training bbox detector")
    for images, labels, bboxes, _ in pbar:
        images = images.to(device)
        bboxes = bboxes.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass (only bbox detector)
        pred_bboxes = model.bbox_detector(images)
        
        # Calculate bbox loss
        loss = bbox_loss_fn(pred_bboxes, bboxes)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'bbox_loss': loss.item()})
    
    return total_loss / len(dataloader)


def train_stage2_epoch(model, dataloader, criterion, optimizer, device):
    """Train Stage 2 (full model) for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Stage 2: Training full model")
    for images, labels, bboxes, _ in pbar:
        images = images.to(device)
        labels = labels.to(device)
        bboxes = bboxes.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass (full model)
        pred_bboxes, char_outputs = model.forward(images, return_bboxes=True)
        
        # Calculate bbox loss (small weight)
        bbox_loss = bbox_loss_fn(pred_bboxes, bboxes) * 0.1
        
        # Calculate character classification loss
        char_loss = 0
        for i, output in enumerate(char_outputs):
            char_loss += criterion(output, labels[:, i])
        char_loss = char_loss / len(char_outputs)
        
        # Total loss
        loss = bbox_loss + char_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        predictions = model.predict(images)
        correct += (predictions == labels).all(dim=1).sum().item()
        total += images.size(0)
        
        total_loss += loss.item()
        pbar.set_postfix({
            'loss': loss.item(), 
            'char_loss': char_loss.item(),
            'bbox_loss': bbox_loss.item(),
            'acc': f'{100*correct/total:.2f}%'
        })
    
    return total_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device):
    """Validate the full model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, bboxes, _ in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)
            bboxes = bboxes.to(device)
            
            pred_bboxes, char_outputs = model.forward(images, return_bboxes=True)
            
            # Calculate loss
            bbox_loss = bbox_loss_fn(pred_bboxes, bboxes) * 0.1
            char_loss = 0
            for i, output in enumerate(char_outputs):
                char_loss += criterion(output, labels[:, i])
            char_loss = char_loss / len(char_outputs)
            loss = bbox_loss + char_loss
            
            # Calculate accuracy
            predictions = model.predict(images)
            correct += (predictions == labels).all(dim=1).sum().item()
            total += images.size(0)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader), correct / total


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if preprocessed data exists
    if not DATA_DIR.exists() or len(list(DATA_DIR.glob("*.png"))) == 0:
        print(f"\n⚠️  ERROR: No preprocessed images found in {DATA_DIR}")
        print(f"Please run preprocessing first:")
        print(f"  python preprocess.py")
        return
    
    # Create transforms for grayscale images
    transform = transforms.Compose([
        transforms.Resize((60, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Single channel
    ])
    
    # Load dataset
    print("Loading dataset...")
    full_dataset = CaptchaDatasetGrayscale(DATA_DIR, CHARACTERS, transform=transform)
    
    # Split into train and validation
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    num_workers = 0 if not torch.cuda.is_available() else 2
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    
    # Create model
    model = TwoStageCaptchaModel(
        num_chars=CAPTCHA_LENGTH,
        num_classes=NUM_CLASSES
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer for character recognition
    criterion = nn.CrossEntropyLoss()
    
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # ===== STAGE 1: Train bbox detector =====
    print("\n" + "="*60)
    print("STAGE 1: Training Bounding Box Detector")
    print("="*60)
    
    optimizer_stage1 = optim.Adam(model.bbox_detector.parameters(), lr=LEARNING_RATE_STAGE1)
    
    for epoch in range(EPOCHS_STAGE1):
        print(f"\nEpoch {epoch+1}/{EPOCHS_STAGE1}")
        bbox_loss = train_stage1_epoch(model, train_loader, optimizer_stage1, device)
        print(f"Bbox Loss: {bbox_loss:.4f}")
    
    print("✓ Stage 1 complete!")
    
    # ===== STAGE 2: Train full model =====
    print("\n" + "="*60)
    print("STAGE 2: Training Full Model (with frozen bbox detector)")
    print("="*60)
    
    # Optionally freeze bbox detector (comment out if you want to fine-tune)
    # for param in model.bbox_detector.parameters():
    #     param.requires_grad = False
    
    optimizer_stage2 = optim.Adam(model.parameters(), lr=LEARNING_RATE_STAGE2)
    
    best_val_acc = 0
    
    for epoch in range(EPOCHS_STAGE2):
        print(f"\nEpoch {epoch+1}/{EPOCHS_STAGE2}")
        
        train_loss, train_acc = train_stage2_epoch(model, train_loader, criterion, optimizer_stage2, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {100*train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {100*val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_stage2.state_dict(),
                'val_acc': val_acc,
                'characters': CHARACTERS,
            }, MODEL_SAVE_PATH)
            print(f"✓ Saved best model (acc: {100*val_acc:.2f}%)")
    
    print(f"\n✓ Training complete!")
    print(f"✓ Best validation accuracy: {100*best_val_acc:.2f}%")
    print(f"✓ Model saved to: {MODEL_SAVE_PATH}")
    print(f"\nNOTE: If accuracy is still low, consider:")
    print(f"  1. Running preprocessing: python preprocess.py")
    print(f"  2. Checking if preprocessed images look correct")
    print(f"  3. Adjusting learning rates or model architecture")


if __name__ == "__main__":
    main()
