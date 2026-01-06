"""
Training script for CAPTCHA breaker.
Run this to train your model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import string
from pathlib import Path
from tqdm import tqdm

from src.dataset import CaptchaDataset
from src.model import CaptchaCNN

# Configuration
DATA_DIR = Path("data/raw")
MODEL_SAVE_PATH = Path("models/captcha_model.pth")
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.8
CAPTCHA_LENGTH = 5

# Character set
CHARACTERS = string.digits + string.ascii_uppercase
NUM_CLASSES = len(CHARACTERS)

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels, _ in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Calculate loss for each character position
        loss = 0
        for i, output in enumerate(outputs):
            loss += criterion(output, labels[:, i])
        loss = loss / len(outputs)  # Normalize by number of positions
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        predictions = model.predict(images)
        correct += (predictions == labels).all(dim=1).sum().item()
        total += images.size(0)
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item(), 'acc': f'{100*correct/total:.2f}%'})
    
    return total_loss / len(dataloader), correct / total

def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            # Calculate loss
            loss = 0
            for i, output in enumerate(outputs):
                loss += criterion(output, labels[:, i])
            
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
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize((60, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                           std=[0.5, 0.5, 0.5])
    ])
    
    # Load dataset
    print("Loading dataset...")
    full_dataset = CaptchaDataset(DATA_DIR, CHARACTERS, transform=transform)
    
    # Split into train and validation
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders (num_workers=0 on macOS/Windows, 2 on Linux with GPU)
    num_workers = 0 if torch.cuda.is_available() == False else 2
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    
    # Create model
    model = CaptchaCNN(
        num_chars=CAPTCHA_LENGTH,
        num_classes=NUM_CLASSES,
        captcha_length=CAPTCHA_LENGTH
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    best_val_acc = 0
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    print("\nStarting training...")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {100*train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {100*val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'characters': CHARACTERS,
            }, MODEL_SAVE_PATH)
            print(f"✓ Saved best model (acc: {100*val_acc:.2f}%)")
    
    print(f"\n✓ Training complete!")
    print(f"✓ Best validation accuracy: {100*best_val_acc:.2f}%")
    print(f"✓ Model saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
