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
    Remove background dots and noise using morphological operations.
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    
    # Apply adaptive thresholding to separate foreground from background
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Morphological opening to remove small noise
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Invert back (characters should be dark on light background)
    cleaned = cv2.bitwise_not(cleaned)
    
    return cleaned


def preprocess_image(input_path, output_path):
    """
    Preprocess a single CAPTCHA image.
    
    Args:
        input_path: Path to input color image
        output_path: Path to save processed grayscale image
    """
    # Read image
    img = cv2.imread(str(input_path))
    
    if img is None:
        print(f"Warning: Could not read {input_path}")
        return False
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Remove background noise
    cleaned = remove_background_noise(gray)
    
    # Optional: Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(cleaned)
    
    # Optional: Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    
    # Save preprocessed image
    cv2.imwrite(str(output_path), denoised)
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Preprocess CAPTCHA images')
    parser.add_argument('--input_dir', type=str, default='data/raw',
                        help='Input directory with raw CAPTCHA images')
    parser.add_argument('--output_dir', type=str, default='data/processed',
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
