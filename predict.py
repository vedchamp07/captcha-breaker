"""
Use trained model to predict CAPTCHA text from images.
"""
import torch
from torchvision import transforms
from PIL import Image
import argparse
from pathlib import Path

from src.model import CaptchaCNN

def load_model(model_path, device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    characters = checkpoint['characters']
    num_classes = len(characters)
    
    model = CaptchaCNN(
        num_chars=5,
        num_classes=num_classes,
        captcha_length=5
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create index to character mapping
    idx_to_char = {idx: char for idx, char in enumerate(characters)}
    
    return model, idx_to_char

def predict_image(image_path, model, idx_to_char, device):
    """Predict CAPTCHA text from image."""
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((60, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        predictions = model.predict(image_tensor)
    
    # Decode prediction
    predicted_text = ''.join([idx_to_char[idx.item()] for idx in predictions[0]])
    
    return predicted_text

def main():
    parser = argparse.ArgumentParser(description='Predict CAPTCHA text')
    parser.add_argument('image_path', type=str, help='Path to CAPTCHA image')
    parser.add_argument('--model', type=str, default='models/captcha_model.pth',
                       help='Path to trained model')
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print(f"Loading model from {args.model}...")
    model, idx_to_char = load_model(args.model, device)
    
    # Predict
    print(f"Predicting {args.image_path}...")
    predicted_text = predict_image(args.image_path, model, idx_to_char, device)
    
    print(f"\n✓ Predicted text: {predicted_text}")
    
    # If filename contains actual text, show accuracy
    filename = Path(args.image_path).stem
    if '_' in filename:
        actual_text = filename.split('_')[0]
        correct = predicted_text == actual_text
        print(f"  Actual text: {actual_text}")
        print(f"  Match: {'✓' if correct else '✗'}")

if __name__ == "__main__":
    main()
