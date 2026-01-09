"""
Generate CAPTCHA dataset using custom fonts from train_font_library.
Creates 100 images per font.
"""
from captcha.image import ImageCaptcha
import random
import string
from pathlib import Path
from tqdm import tqdm

# Configuration
OUTPUT_DIR = Path("data/font_test/raw")
FONT_DIR = Path("train_font_library")
CAPTCHA_LENGTH = 5
IMAGES_PER_FONT = 100
CHARACTERS = string.digits + string.ascii_uppercase

def find_font_files():
    """Find all TTF font files in the font directory."""
    fonts = []
    for font_path in FONT_DIR.rglob("*.ttf"):
        # Skip static/variable font variants, prefer main ones
        if "static" not in str(font_path):
            fonts.append(font_path)
    return sorted(fonts)

def generate_captcha_text(length):
    """Generate random CAPTCHA text."""
    return ''.join(random.choices(CHARACTERS, k=length))

def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all fonts
    font_files = find_font_files()
    
    if not font_files:
        print("No font files found in train_font_library/")
        return
    
    print(f"Found {len(font_files)} fonts")
    print(f"Generating {IMAGES_PER_FONT} images per font")
    print(f"Total images: {len(font_files) * IMAGES_PER_FONT}")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    total_generated = 0
    
    for font_path in tqdm(font_files, desc="Processing fonts"):
        font_name = font_path.stem
        
        # Create ImageCaptcha with specific font
        image = ImageCaptcha(fonts=[str(font_path)])
        
        for i in range(IMAGES_PER_FONT):
            # Generate random text
            text = generate_captcha_text(CAPTCHA_LENGTH)
            
            # Create filename with font name prefix
            filename = f"{text}_{font_name}_{i}.png"
            filepath = OUTPUT_DIR / filename
            
            # Generate and save image
            image.write(text, str(filepath))
            total_generated += 1
    
    print(f"\n✓ Successfully generated {total_generated} CAPTCHA images!")
    print(f"✓ Saved to: {OUTPUT_DIR}")
    print(f"✓ {len(font_files)} fonts used")

if __name__ == "__main__":
    main()
