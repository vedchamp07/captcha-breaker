"""
Preprocessing script for CAPTCHA images.
Converts to grayscale, removes noise, and applies enhancements.
"""
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def remove_background_noise(image):
    """
    Remove small background dots while preserving text characters.
    Returns grayscale text on white background.
    """
    # Apply light Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    
    # Use Otsu's thresholding to separate foreground (text) from background
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invert if needed (text should be dark on light background)
    # Check if most pixels are dark - if so, we need to invert
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)
    
    # Remove small noise dots using morphological opening
    # Small kernel to remove dots but preserve text structure
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Close small gaps in text
    kernel_close = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    return cleaned


def preprocess_image(input_path, output_path):
    """
    Preprocess a single CAPTCHA image.
    Keeps original resolution, outputs grayscale text on white background.
    
    Args:
        input_path: Path to input color image
        output_path: Path to save processed grayscale image
    """
    # Read image at original resolution
    img = cv2.imread(str(input_path))
    
    if img is None:
        print(f"Warning: Could not read {input_path}")
        return False
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Remove small background noise dots
    cleaned = remove_background_noise(gray)
    
    # Save preprocessed image (grayscale text on white background)
    cv2.imwrite(str(output_path), cleaned)
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Preprocess CAPTCHA images')
    parser.add_argument('--input_dir', type=str, default='data/train/raw',
                        help='Input directory with raw CAPTCHA images')
    parser.add_argument('--output_dir', type=str, default='data/train/processed',
                        help='Output directory for processed images')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(input_dir.glob("*.png"))
    
    print(f"Found {len(image_files)} images to preprocess")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Process each image
    successful = 0
    failed = 0
    
    for img_path in tqdm(image_files, desc="Preprocessing"):
        output_path = output_dir / img_path.name
        
        if preprocess_image(img_path, output_path):
            successful += 1
        else:
            failed += 1
    
    print(f"\n✓ Preprocessing complete!")
    print(f"✓ Successfully processed: {successful} images")
    if failed > 0:
        print(f"✗ Failed: {failed} images")
    print(f"✓ Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
