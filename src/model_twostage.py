"""
Two-stage CAPTCHA recognition model.
Stage 1: Bounding box detection for each character
Stage 2: Character recognition in normalized boxes
"""
import torch
import torch.nn as nn
import torchvision.models as models


class BoundingBoxDetector(nn.Module):
    """
    Stage 1: Detects bounding boxes for each character in CAPTCHA.
    Outputs 5 boxes (x, y, w, h, angle) for each character.
    """
    
    def __init__(self, num_chars=5):
        super(BoundingBoxDetector, self).__init__()
        
        self.num_chars = num_chars
        
        # Convolutional backbone
        self.features = nn.Sequential(
            # Input: 1 x 60 x 160 (grayscale)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32 x 30 x 80
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64 x 15 x 40
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128 x 7 x 20
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 256 x 3 x 10
        )
        
        # Calculate flattened size
        self.fc_input_size = 256 * 3 * 10
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # Output: 5 bounding boxes, each with (x, y, w, h, angle)
        # Total: num_chars * 5 parameters
        self.bbox_head = nn.Linear(512, num_chars * 5)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 1, 60, 160)
        Returns:
            Bounding boxes of shape (batch_size, num_chars, 5)
            Each box: [x_center, y_center, width, height, angle]
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        bbox_flat = self.bbox_head(x)
        
        # Reshape to (batch_size, num_chars, 5)
        bboxes = bbox_flat.view(-1, self.num_chars, 5)
        
        # Apply sigmoid to normalize coordinates to [0, 1]
        # x, y, w, h should be in [0, 1], angle in [-1, 1] (representing -45 to 45 degrees)
        bboxes[..., :4] = torch.sigmoid(bboxes[..., :4])
        bboxes[..., 4] = torch.tanh(bboxes[..., 4])
        
        return bboxes


class CharacterRecognizer(nn.Module):
    """
    Stage 2: Recognizes individual characters from cropped/rotated boxes.
    """
    
    def __init__(self, num_classes=36, input_size=(40, 40)):
        """
        Args:
            num_classes: Number of character classes (36 for A-Z, 0-9)
            input_size: Size of normalized character patches
        """
        super(CharacterRecognizer, self).__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Convolutional layers for character recognition
        self.features = nn.Sequential(
            # Input: 1 x 40 x 40
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32 x 20 x 20
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64 x 10 x 10
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128 x 5 x 5
        )
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(128 * 5 * 5, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 1, 40, 40)
        Returns:
            Class logits of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class TwoStageCaptchaModel(nn.Module):
    """
    Complete two-stage CAPTCHA recognition model.
    """
    
    def __init__(self, num_chars=5, num_classes=36):
        super(TwoStageCaptchaModel, self).__init__()
        
        self.num_chars = num_chars
        self.num_classes = num_classes
        
        # Stage 1: Bounding box detector
        self.bbox_detector = BoundingBoxDetector(num_chars=num_chars)
        
        # Stage 2: Character recognizer
        self.char_recognizer = CharacterRecognizer(num_classes=num_classes)
    
    def extract_character_patches(self, images, bboxes):
        """
        Extract and normalize character patches from images using bounding boxes.
        
        Args:
            images: (batch_size, 1, H, W)
            bboxes: (batch_size, num_chars, 5) - [x, y, w, h, angle]
        
        Returns:
            patches: (batch_size * num_chars, 1, 40, 40)
        """
        batch_size = images.size(0)
        H, W = images.size(2), images.size(3)
        patches = []
        
        for b in range(batch_size):
            for c in range(self.num_chars):
                # Get bbox parameters
                x, y, w, h, angle = bboxes[b, c]
                
                # Convert normalized coordinates to pixel coordinates
                cx = int(x.item() * W)
                cy = int(y.item() * H)
                bw = int(w.item() * W)
                bh = int(h.item() * H)
                
                # Ensure valid crop region
                x1 = max(0, cx - bw // 2)
                y1 = max(0, cy - bh // 2)
                x2 = min(W, cx + bw // 2)
                y2 = min(H, cy + bh // 2)
                
                # Crop patch
                if x2 > x1 and y2 > y1:
                    patch = images[b:b+1, :, y1:y2, x1:x2]
                    # Resize to fixed size
                    patch = torch.nn.functional.interpolate(
                        patch, size=(40, 40), mode='bilinear', align_corners=False
                    )
                else:
                    # Invalid box, create blank patch
                    patch = torch.zeros(1, 1, 40, 40, device=images.device)
                
                patches.append(patch)
        
        return torch.cat(patches, dim=0)
    
    def forward(self, x, return_bboxes=False):
        """
        Args:
            x: Input images (batch_size, 1, 60, 160)
            return_bboxes: Whether to return bounding boxes
        
        Returns:
            If return_bboxes=False:
                List of character logits, one per character position
            If return_bboxes=True:
                (bboxes, character_logits_list)
        """
        # Stage 1: Detect bounding boxes
        bboxes = self.bbox_detector(x)
        
        # Extract character patches
        patches = self.extract_character_patches(x, bboxes)
        
        # Stage 2: Recognize characters
        # Process all patches at once
        char_logits = self.char_recognizer(patches)
        
        # Reshape to (batch_size, num_chars, num_classes)
        batch_size = x.size(0)
        char_logits = char_logits.view(batch_size, self.num_chars, self.num_classes)
        
        # Convert to list format (for compatibility with existing training code)
        char_logits_list = [char_logits[:, i, :] for i in range(self.num_chars)]
        
        if return_bboxes:
            return bboxes, char_logits_list
        else:
            return char_logits_list
    
    def predict(self, x):
        """Get predicted characters from input images."""
        self.eval()
        with torch.no_grad():
            char_logits_list = self.forward(x, return_bboxes=False)
            predictions = [torch.argmax(logits, dim=1) for logits in char_logits_list]
            return torch.stack(predictions, dim=1)  # (batch_size, num_chars)
