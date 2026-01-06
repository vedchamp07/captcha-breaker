"""
PyTorch Dataset class for CAPTCHA images.
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from pathlib import Path

class CaptchaDataset(Dataset):
    """Dataset for CAPTCHA images."""
    
    def __init__(self, data_dir, characters, transform=None):
        """
        Args:
            data_dir: Directory containing captcha images
            characters: String of all possible characters (e.g., "0123456789ABCD...")
            transform: Optional transform to apply to images
        """
        self.data_dir = Path(data_dir)
        self.characters = characters
        self.transform = transform
        
        # Create character to index mapping
        self.char_to_idx = {char: idx for idx, char in enumerate(characters)}
        self.idx_to_char = {idx: char for idx, char in enumerate(characters)}
        
        # Get all image files
        self.image_files = sorted(list(self.data_dir.glob("*.png")))
        
        print(f"Found {len(self.image_files)} images")
        print(f"Character set size: {len(self.characters)}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Get image path
        img_path = self.image_files[idx]
        
        # Extract label from filename (format: "ABC12_0.png")
        filename = img_path.stem  # "ABC12_0"
        label_text = filename.split('_')[0]  # "ABC12"
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Convert label to indices
        label = torch.tensor([self.char_to_idx[c] for c in label_text], 
                            dtype=torch.long)
        
        return image, label, label_text
    
    def decode_prediction(self, indices):
        """Convert predicted indices back to text."""
        return ''.join([self.idx_to_char[idx] for idx in indices])
