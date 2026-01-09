"""
Preprocess font test dataset images to grayscale.
"""
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Configuration
INPUT_DIR = Path("data/font_test/raw")
OUTPUT_DIR = Path("data/font_test/processed")

def preprocess_image(image_path, output_path):
    """Convert image to grayscale and apply basic preprocessing."""
    # Load image
    img = Image.open(image_path)
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Convert to numpy array for OpenCV processing
    img_array = np.array(img)
    
    # Apply Otsu's thresholding
    _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Denoise
    denoised = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    
    # Convert back to PIL Image
    result = Image.fromarray(denoised)
    
    # Save
    result.save(output_path)

def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all PNG files
    image_files = list(INPUT_DIR.glob("*.png"))
    
    if not image_files:
        print(f"No images found in {INPUT_DIR}")
        return
    
    print(f"Found {len(image_files)} images to preprocess")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Process all images
    for img_path in tqdm(image_files, desc="Preprocessing"):
        output_path = OUTPUT_DIR / img_path.name
        preprocess_image(img_path, output_path)
    
    print(f"\n✓ Preprocessing complete!")
    print(f"✓ Successfully processed: {len(image_files)} images")
    print(f"✓ Output saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
