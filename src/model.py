"""
CTC-based CAPTCHA recognition model.
Uses CNN + LSTM + CTC loss - no bounding boxes needed!

This approach is standard for sequence recognition tasks where
character positions are unknown or variable.
"""
import torch
import torch.nn as nn


class CTCCaptchaModel(nn.Module):
    """
    CAPTCHA recognition using CTC (Connectionist Temporal Classification).
    
    Architecture:
    1. CNN backbone extracts visual features
    2. Reshape to sequence (treating width as time steps)
    3. Bidirectional LSTM processes sequence
    4. Linear layer outputs character probabilities for each time step
    5. CTC loss handles alignment between predictions and ground truth
    
    No need for bounding boxes - CTC figures out alignment automatically!
    """
    
    def __init__(self, num_classes=36, hidden_size=256, num_lstm_layers=2, use_attention=False):
        """
        Args:
            num_classes: Number of character classes (36 for A-Z, 0-9)
            hidden_size: Hidden size for LSTM layers
            num_lstm_layers: Number of LSTM layers
        """
        super(CTCCaptchaModel, self).__init__()
        
        self.num_classes = num_classes
        # CTC needs blank token for alignment (class index = num_classes)
        self.blank_idx = num_classes
        
        # CNN backbone for feature extraction
        # Input: (batch, 1, 60, 160) - grayscale image
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> (32, 30, 80)
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> (64, 15, 40)
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),  # Pool only width -> (128, 15, 20)
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),  # Pool only width -> (256, 15, 10)
        )
        
        # After CNN: (batch, 256, 15, 10)
        # We'll reshape to: (batch, 10, 256*15) treating width as sequence
        # So sequence length = 10, feature dim = 256*15 = 3840
        self.feature_size = 256 * 15  # channels * height
        self.sequence_length = 10  # width after pooling
        
        # Map CNN features to LSTM input size
        self.map_to_seq = nn.Linear(self.feature_size, hidden_size)
        
        # Bidirectional LSTM to process sequence
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=True,
            dropout=0.3 if num_lstm_layers > 1 else 0,
            batch_first=True
        )

        # Optional self-attention on top of LSTM outputs
        self.use_attention = use_attention
        if self.use_attention:
            self.attn = nn.MultiheadAttention(hidden_size * 2, num_heads=4, dropout=0.1, batch_first=True)
            self.attn_norm = nn.LayerNorm(hidden_size * 2)
            self.attn_dropout = nn.Dropout(0.1)
        else:
            self.attn = None
        
        # Output layer: map LSTM outputs to character probabilities
        # +1 for CTC blank token
        self.fc = nn.Linear(hidden_size * 2, num_classes + 1)  # *2 for bidirectional
        
    def forward(self, x):
        """
        Args:
            x: Input images (batch_size, 1, 60, 160)
        
        Returns:
            Log probabilities for CTC loss (sequence_length, batch_size, num_classes+1)
        """
        batch_size = x.size(0)
        
        # Extract CNN features
        features = self.cnn(x)  # (batch, 256, 15, 10)
        
        # Reshape to sequence: (batch, width, channels*height)
        # Transpose to treat width as sequence dimension
        features = features.permute(0, 3, 1, 2)  # (batch, 10, 256, 15)
        features = features.reshape(batch_size, self.sequence_length, self.feature_size)
        
        # Map to LSTM input size
        features = self.map_to_seq(features)  # (batch, 10, hidden_size)
        
        # Process with LSTM
        lstm_out, _ = self.lstm(features)  # (batch, 10, hidden_size*2)
        
        # Optional attention
        if self.attn is not None:
            attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
            lstm_out = self.attn_norm(lstm_out + self.attn_dropout(attn_out))

        # Get character predictions for each time step
        logits = self.fc(lstm_out)  # (batch, 10, num_classes+1)
        
        # CTC expects: (sequence_length, batch, num_classes)
        logits = logits.permute(1, 0, 2)  # (10, batch, num_classes+1)
        
        # Apply log_softmax for CTC loss
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)
        
        return log_probs
    
    def predict(self, x):
        """
        Decode predictions using greedy decoding.
        
        Args:
            x: Input images (batch_size, 1, 60, 160)
        
        Returns:
            Predicted character indices (batch_size, max_length)
        """
        self.eval()
        with torch.no_grad():
            log_probs = self.forward(x)  # (seq_len, batch, num_classes+1)
            
            # Greedy decoding: take argmax at each time step
            _, preds = log_probs.max(2)  # (seq_len, batch)
            preds = preds.transpose(0, 1)  # (batch, seq_len)
            
            # Decode: remove blanks and repeated characters
            decoded = []
            for pred_seq in preds:
                decoded_seq = []
                prev_char = None
                
                for char_idx in pred_seq:
                    char_idx = char_idx.item()
                    
                    # Skip blank tokens
                    if char_idx == self.blank_idx:
                        prev_char = None
                        continue
                    
                    # Skip repeated characters (CTC rule)
                    if char_idx != prev_char:
                        decoded_seq.append(char_idx)
                        prev_char = char_idx
                
                decoded.append(decoded_seq)
            
            # Pad sequences to same length (max 5 for CAPTCHA)
            max_len = 5
            padded = []
            for seq in decoded:
                if len(seq) < max_len:
                    seq = seq + [0] * (max_len - len(seq))  # Pad with 0
                else:
                    seq = seq[:max_len]  # Truncate if too long
                padded.append(seq)
            
            # Return tensor on same device as input
            return torch.tensor(padded, dtype=torch.long, device=x.device)


class CTCCaptchaModelSimple(nn.Module):
    """
    Simpler CTC model without LSTM (faster training, less memory).
    Good baseline to start with.
    """
    
    def __init__(self, num_classes=36):
        super(CTCCaptchaModelSimple, self).__init__()
        
        self.num_classes = num_classes
        self.blank_idx = num_classes
        
        # CNN backbone
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # -> (64, 30, 80)
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # -> (128, 15, 40)
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),  # -> (256, 15, 20)
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),  # -> (512, 15, 10)
        )
        
        # Direct mapping to character predictions
        # Treat width dimension as sequence
        self.classifier = nn.Sequential(
            nn.Linear(512 * 15, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes + 1)
        )
        
        self.sequence_length = 10
        
    def forward(self, x):
        """Forward pass for CTC."""
        batch_size = x.size(0)
        
        # Extract features
        features = self.features(x)  # (batch, 512, 15, 10)
        
        # Reshape: treat width as sequence
        features = features.permute(0, 3, 1, 2)  # (batch, 10, 512, 15)
        features = features.reshape(batch_size, self.sequence_length, -1)
        
        # Classify each time step
        logits = self.classifier(features)  # (batch, 10, num_classes+1)
        
        # CTC format
        logits = logits.permute(1, 0, 2)  # (10, batch, num_classes+1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)
        
        return log_probs
    
    def predict(self, x):
        """Greedy decoding."""
        self.eval()
        with torch.no_grad():
            log_probs = self.forward(x)
            _, preds = log_probs.max(2)
            preds = preds.transpose(0, 1)
            
            # Decode
            decoded = []
            for pred_seq in preds:
                decoded_seq = []
                prev_char = None
                
                for char_idx in pred_seq:
                    char_idx = char_idx.item()
                    if char_idx == self.blank_idx:
                        prev_char = None
                        continue
                    if char_idx != prev_char:
                        decoded_seq.append(char_idx)
                        prev_char = char_idx
                
                decoded.append(decoded_seq)
            
            # Pad to length 5
            max_len = 5
            padded = []
            for seq in decoded:
                if len(seq) < max_len:
                    seq = seq + [0] * (max_len - len(seq))
                else:
                    seq = seq[:max_len]
                padded.append(seq)
            
            # Return tensor on same device as input
            return torch.tensor(padded, dtype=torch.long, device=x.device)
