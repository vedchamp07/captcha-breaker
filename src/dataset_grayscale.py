"""
PyTorch Dataset class for preprocessed grayscale CAPTCHA images.
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from pathlib import Path
import numpy as np


class CaptchaDatasetGrayscale(Dataset):
    """Dataset for grayscale preprocessed CAPTCHA images."""
    
    def __init__(self, data_dir, characters, transform=None):
        """
        Args:
            data_dir: Directory containing preprocessed captcha images
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
        
        # Load image as grayscale
        image = Image.open(img_path).convert('L')  # 'L' mode for grayscale
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Convert label to indices
        label = torch.tensor([self.char_to_idx[c] for c in label_text], 
                            dtype=torch.long)
        
        # Generate dummy bounding boxes (for training stage 1)
        # In practice, you'd need ground truth boxes. For now, we estimate based on position
        bboxes = self.generate_approximate_bboxes(label_text)
        
        return image, label, bboxes, label_text
    
    def generate_approximate_bboxes(self, label_text):
        """
        Generate approximate bounding boxes for characters.
        This is a rough estimate. For best results, you'd need annotated data.
        
        Format: [x_center, y_center, width, height, angle]
        All normalized to [0, 1] except angle which is in radians [-pi/4, pi/4]
        """
        num_chars = len(label_text)
        bboxes = []
        
        # Assume characters are roughly evenly spaced
        # Image width = 160, height = 60 (from your config)
        for i in range(num_chars):
            # Approximate x position (evenly distributed)
            x = (i + 0.5) / num_chars
            y = 0.5  # Centered vertically
            w = 0.15  # Approximate character width
            h = 0.6   # Approximate character height
            angle = 0.0  # Assume no rotation for synthetic CAPTCHAs
            
            bboxes.append([x, y, w, h, angle])
        
        return torch.tensor(bboxes, dtype=torch.float32)
    
    def decode_prediction(self, indices):
        """Convert predicted indices back to text."""
        return ''.join([self.idx_to_char[idx] for idx in indices])
