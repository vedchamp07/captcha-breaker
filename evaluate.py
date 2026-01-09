"""
Batch evaluation script for CTC-based CAPTCHA model.
Evaluates model accuracy on multiple CAPTCHA images.
"""
import torch
from torchvision import transforms
from PIL import Image
import string
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
import cv2

from src.model import CTCCaptchaModel, CTCCaptchaModelSimple


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


def predict_image(model, image_path, characters, device):
    """Predict CAPTCHA text from image."""
    # Load and preprocess image
    image = Image.open(image_path).convert('L')
    image = preprocess_image(image)
    
    transform = transforms.Compose([
        transforms.Resize((60, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        pred_indices = model.predict(image_tensor)[0]
    
    # Decode
    predicted_text = ''.join([characters[idx] for idx in pred_indices if idx < len(characters)])
    
    return predicted_text


def evaluate_batch(model_path, data_dir, characters, device, use_attention=False, max_samples=None, use_lstm=True):
    """Evaluate model on batch of images."""
    # Load model
    if use_lstm:
        model = CTCCaptchaModel(num_classes=len(characters), use_attention=use_attention)
    else:
        model = CTCCaptchaModelSimple(num_classes=len(characters))
    
    # Load checkpoint - handle both direct state_dict and checkpoint format
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Checkpoint format with metadata
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Direct state_dict
        model.load_state_dict(checkpoint)
    
    model.to(device)
    
    # Get all images
    data_path = Path(data_dir)
    image_files = sorted(list(data_path.glob("*.png")))
    
    if max_samples:
        image_files = image_files[:max_samples]
    
    print(f"\nModel: {model_path}")
    print(f"Device: {device}")
    print(f"Character Set: {len(characters)} chars ({characters[:10]}...{characters[-10:]})")
    print(f"Evaluating on {len(image_files)} images...\n")
    
    correct = 0
    total = 0
    incorrect_samples = []
    font_stats = {}  # Track per-font statistics
    
    for image_path in tqdm(image_files, desc="Evaluating"):
        # Get ground truth from filename
        if '_' not in image_path.stem:
            continue
        
        parts = image_path.stem.split('_')
        ground_truth = parts[0]
        
        # Extract font name if available (format: TEXT_FONT_INDEX.png)
        font_name = parts[1] if len(parts) > 1 else "unknown"
        
        # Initialize font stats if not seen before
        if font_name not in font_stats:
            font_stats[font_name] = {'correct': 0, 'total': 0}
        
        # Predict
        predicted_text = predict_image(model, image_path, characters, device)
        
        # Check correctness
        is_correct = predicted_text == ground_truth
        if is_correct:
            correct += 1
            font_stats[font_name]['correct'] += 1
        else:
            incorrect_samples.append({
                'filename': image_path.name,
                'predicted': predicted_text,
                'ground_truth': ground_truth
            })
        
        total += 1
        font_stats[font_name]['total'] += 1
    
    # Print results
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Overall Results:")
    print(f"{'='*60}")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'='*60}\n")
    
    # Print font-wise accuracy if multiple fonts detected
    if len(font_stats) > 1:
        print(f"{'='*60}")
        print(f"Font-wise Accuracy:")
        print(f"{'='*60}")
        
        # Sort fonts by accuracy
        sorted_fonts = sorted(font_stats.items(), 
                            key=lambda x: x[1]['correct']/x[1]['total'] if x[1]['total'] > 0 else 0,
                            reverse=True)
        
        for font_name, stats in sorted_fonts:
            font_acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"{font_name:40s} {stats['correct']:4d}/{stats['total']:4d}  ({font_acc:5.2f}%)")
        
        print(f"{'='*60}\n")
    
    # Print first few incorrect samples if any
    if incorrect_samples:
        print(f"Sample Incorrect Predictions (showing first 10):\n")
        for sample in incorrect_samples[:10]:
            print(f"  {sample['filename']}")
            print(f"    Predicted:    {sample['predicted']}")
            print(f"    Ground Truth: {sample['ground_truth']}\n")
    
    return accuracy, correct, total


def main():
    parser = argparse.ArgumentParser(description='Evaluate CAPTCHA model on batch of images')
    parser.add_argument('--model', type=str, default='models/captcha_model_v3.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/test/raw',
                       help='Directory containing CAPTCHA images')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (default: all)')
    parser.add_argument('--use-lstm', action='store_true', default=True,
                       help='Use LSTM model (default: True)')
    parser.add_argument('--use-attention', action='store_true', default=True,
                       help='Enable self-attention in LSTM model (default: True)')
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    characters = string.digits + string.ascii_lowercase + string.ascii_uppercase
    
    # Evaluate
    accuracy, correct, total = evaluate_batch(
        args.model,
        args.data_dir,
        characters,
        device,
        use_attention=args.use_attention,
        max_samples=args.max_samples,
        use_lstm=args.use_lstm
    )


if __name__ == "__main__":
    main()
