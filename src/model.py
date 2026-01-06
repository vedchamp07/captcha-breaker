"""
CNN-based CAPTCHA recognition model.
"""
import torch
import torch.nn as nn

class CaptchaCNN(nn.Module):
    """Convolutional Neural Network for CAPTCHA recognition."""
    
    def __init__(self, num_chars, num_classes, captcha_length=5):
        """
        Args:
            num_chars: Number of character positions to predict (e.g., 5)
            num_classes: Number of possible characters (e.g., 36 for 0-9,A-Z)
            captcha_length: Length of CAPTCHA text
        """
        super(CaptchaCNN, self).__init__()
        
        self.num_chars = num_chars
        self.num_classes = num_classes
        self.captcha_length = captcha_length
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Input: 3 x 60 x 160
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32 x 30 x 80
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64 x 15 x 40
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128 x 7 x 20
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 256 x 3 x 10
        )
        
        # Calculate flattened size: 256 * 3 * 10 = 7680
        self.fc_input_size = 256 * 3 * 10
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        # Output layers: one for each character position
        self.output_layers = nn.ModuleList([
            nn.Linear(512, num_classes) for _ in range(captcha_length)
        ])
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
        Returns:
            List of tensors, one for each character position
            Each tensor has shape (batch_size, num_classes)
        """
        # Convolutional layers
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc_layers(x)
        
        # Predict each character position
        outputs = [layer(x) for layer in self.output_layers]
        
        return outputs
    
    def predict(self, x):
        """Get predicted characters from logits."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            predictions = [torch.argmax(out, dim=1) for out in outputs]
            return torch.stack(predictions, dim=1)  # (batch_size, captcha_length)
