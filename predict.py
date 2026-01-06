"""
Prediction script for CTC-based CAPTCHA model.
"""
import torch
from torchvision import transforms
from PIL import Image
import string
from pathlib import Path
import argparse

from src.model import CTCCaptchaModel, CTCCaptchaModelSimple


def predict_image(model, image_path, characters, device):
    """Predict CAPTCHA text from image."""
    # Handle relative/absolute paths
    image_path = Path(image_path)
    
    # If relative path and doesn't exist, try relative to script
    if not image_path.is_absolute() and not image_path.exists():
        alt_path = Path(__file__).parent / image_path
        if alt_path.exists():
            image_path = alt_path
    
    # Final check
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load and preprocess image
    image = Image.open(image_path).convert('L')
    
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


def main():
    parser = argparse.ArgumentParser(description='Predict CAPTCHA using CTC model')
    parser.add_argument('image_path', type=str, help='Path to CAPTCHA image')
    parser.add_argument('--model', type=str, default='models/captcha_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--use-lstm', action='store_true', default=True,
                       help='Use LSTM model (default: True)')
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    characters = string.digits + string.ascii_uppercase
    
    # Load model
    if args.use_lstm:
        model = CTCCaptchaModel(num_classes=len(characters))
    else:
        model = CTCCaptchaModelSimple(num_classes=len(characters))
    
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    
    # Predict
    predicted_text = predict_image(model, args.image_path, characters, device)
    
    # Get ground truth if available
    image_path = Path(args.image_path)
    if '_' in image_path.stem:
        ground_truth = image_path.stem.split('_')[0]
        correct = predicted_text == ground_truth
        print(f"Predicted: {predicted_text}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Correct: {'✓' if correct else '✗'}")
    else:
        print(f"Predicted: {predicted_text}")


if __name__ == "__main__":
    main()
