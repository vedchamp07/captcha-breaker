"""
Analyze character confusion in model predictions.
Helps identify which characters are most often confused.
"""
import torch
from torchvision import transforms
from PIL import Image
import string
from pathlib import Path
import argparse
import numpy as np
from collections import defaultdict
import cv2

from src.model import CTCCaptchaModel


def preprocess_image(image):
    """Preprocess image: grayscale, thresholding, morphological operations."""
    img_array = np.array(image.convert('L'))

    # Invert if background is dark to get dark text on light background
    if img_array.mean() < 127:
        img_array = 255 - img_array

    _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return Image.fromarray(processed)


def analyze_confusion(model_path, data_dir, device):
    """Analyze character-level confusion in predictions."""
    
    characters = string.digits + string.ascii_lowercase + string.ascii_uppercase
    
    # Load model
    model = CTCCaptchaModel(num_classes=len(characters), use_attention=True)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((60, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Confusion matrix: confusion[truth][prediction] = count
    confusion = defaultdict(lambda: defaultdict(int))
    total_chars = defaultdict(int)
    
    # Get all images
    data_path = Path(data_dir)
    image_files = sorted(list(data_path.glob("*.png")))
    
    print(f"Analyzing {len(image_files)} images...\n")
    
    for img_path in image_files:
        # Get ground truth
        ground_truth = img_path.stem.split('_')[0]
        
        # Load and predict
        image = Image.open(img_path).convert('L')
        image = preprocess_image(image)
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            decoded = model.predict(image_tensor)

        pred_indices = decoded[0] if decoded else []
        predicted_text = ''.join([characters[idx] for idx in pred_indices if 0 <= idx < len(characters)])
        
        # Align and count character-level confusions
        # Simple alignment: compare position by position
        max_len = max(len(ground_truth), len(predicted_text))
        
        for i in range(max_len):
            truth_char = ground_truth[i] if i < len(ground_truth) else '_'
            pred_char = predicted_text[i] if i < len(predicted_text) else '_'
            
            if truth_char != '_':
                total_chars[truth_char] += 1
                confusion[truth_char][pred_char] += 1
    
    # Print confusion matrix for commonly confused pairs
    print("=" * 80)
    print("CHARACTER CONFUSION ANALYSIS")
    print("=" * 80)
    
    confusing_pairs = [
        ('0', 'O'),
        ('1', 'I', 'l'),
        ('5', 'S'),
        ('8', 'B'),
        ('6', 'b'),
        ('2', 'Z'),
    ]
    
    print("\nKnown Confusing Character Groups:")
    print("-" * 80)
    
    for group in confusing_pairs:
        print(f"\nGroup: {', '.join(group)}")
        for char in group:
            if char in confusion:
                total = total_chars[char]
                print(f"  '{char}' (n={total}):")
                # Sort by confusion count
                sorted_confusions = sorted(confusion[char].items(), 
                                         key=lambda x: x[1], reverse=True)[:5]
                for pred_char, count in sorted_confusions:
                    pct = 100 * count / total if total > 0 else 0
                    marker = " ⚠️ " if pred_char in group and pred_char != char else ""
                    print(f"    → '{pred_char}': {count:4d} ({pct:5.1f}%){marker}")
    
    # Print top overall confusions
    print("\n" + "=" * 80)
    print("TOP 20 CONFUSION PAIRS (excluding correct predictions)")
    print("=" * 80)
    
    all_confusions = []
    for truth_char, pred_dict in confusion.items():
        for pred_char, count in pred_dict.items():
            if truth_char != pred_char:  # Exclude correct predictions
                all_confusions.append((truth_char, pred_char, count))
    
    all_confusions.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\n{'Truth':^8} {'Pred':^8} {'Count':^8} {'Percentage':^12}")
    print("-" * 40)
    for truth_char, pred_char, count in all_confusions[:20]:
        total = total_chars[truth_char]
        pct = 100 * count / total if total > 0 else 0
        print(f"  '{truth_char}' {' ':>4} → '{pred_char}' {' ':>4} {count:6d} {' ':>6} {pct:5.1f}%")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Analyze character confusion')
    parser.add_argument('--model', type=str, default='models/captcha_model_v4.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/test/raw',
                       help='Directory containing test images')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    analyze_confusion(args.model, args.data_dir, device)


if __name__ == "__main__":
    main()
