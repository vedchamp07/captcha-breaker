"""
Generate CAPTCHA dataset using python-captcha library.
Run this first to create your training data.
"""
from captcha.image import ImageCaptcha
import random
import string
import os
from pathlib import Path

# Configuration
NUM_SAMPLES = 10000  # Number of captchas to generate
CAPTCHA_LENGTH = 5   # Length of each captcha text
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 60
OUTPUT_DIR = Path("data/train/raw")

# Character set (digits + uppercase letters)
CHARACTERS = string.digits + string.ascii_uppercase

def generate_captcha_text(length=CAPTCHA_LENGTH):
    """Generate random captcha text."""
    return ''.join(random.choices(CHARACTERS, k=length))

def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize captcha generator
    generator = ImageCaptcha(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
    
    print(f"Generating {NUM_SAMPLES} CAPTCHA images...")
    print(f"Characters used: {CHARACTERS}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    for i in range(NUM_SAMPLES):
        # Generate random text
        captcha_text = generate_captcha_text()
        
        # Generate image
        image = generator.generate(captcha_text)
        
        # Save with filename as the captcha text
        filename = f"{captcha_text}_{i}.png"
        filepath = OUTPUT_DIR / filename
        
        generator.write(captcha_text, str(filepath))
        
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1}/{NUM_SAMPLES} images...")
    
    print(f"\nSuccessfully generated {NUM_SAMPLES} CAPTCHA images!")
    print(f"Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
