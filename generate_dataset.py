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
MIN_LENGTH = 3       # Minimum captcha text length
MAX_LENGTH = 7       # Maximum captcha text length
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 60
OUTPUT_DIR = Path("data/train/raw")

# Character set (digits + lowercase + uppercase letters)
CHARACTERS = string.digits + string.ascii_lowercase + string.ascii_uppercase

# Confusing character pairs that need more training examples
CONFUSING_PAIRS = [
    ('0', 'O'),  # Zero and uppercase O
    ('1', 'I', 'l'),  # One, uppercase I, lowercase L
    ('5', 'S'),  # Five and uppercase S
    ('8', 'B'),  # Eight and uppercase B
    ('6', 'b'),  # Six and lowercase b
    ('2', 'Z'),  # Two and uppercase Z
]

def generate_captcha_text(force_confusing=False):
    """Generate random captcha text with variable length.
    
    Args:
        force_confusing: If True, ensure at least one confusing character is included
    """
    length = random.randint(MIN_LENGTH, MAX_LENGTH)
    
    if force_confusing and random.random() < 0.3:  # 30% chance to force confusing chars
        # Pick a random confusing pair and include at least one character from it
        confusing_group = random.choice(CONFUSING_PAIRS)
        text = list(random.choices(CHARACTERS, k=length))
        # Replace random position(s) with confusing characters
        num_replacements = random.randint(1, min(2, length))
        for _ in range(num_replacements):
            pos = random.randint(0, length - 1)
            text[pos] = random.choice(confusing_group)
        return ''.join(text)
    else:
        return ''.join(random.choices(CHARACTERS, k=length))

def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize captcha generator
    generator = ImageCaptcha(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
    
    print(f"Generating {NUM_SAMPLES} CAPTCHA images...")
    print(f"Length range: {MIN_LENGTH}-{MAX_LENGTH} characters")
    print(f"Characters used: {CHARACTERS}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Generate 40% with forced confusing characters
    num_confusing = int(NUM_SAMPLES * 0.4)
    
    for i in range(NUM_SAMPLES):
        # Generate random text (force confusing chars for first 40% of samples)
        force_confusing = i < num_confusing
        captcha_text = generate_captcha_text(force_confusing=force_confusing)
        
        # Generate image
        image = generator.generate(captcha_text)
        
        # Save with filename as the captcha text
        filename = f"{captcha_text}_{i}.png"
        filepath = OUTPUT_DIR / filename
        
        generator.write(captcha_text, str(filepath))
        
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1}/{NUM_SAMPLES} images... (confusing: {min(i+1, num_confusing)})")
    
    print(f"\nSuccessfully generated {NUM_SAMPLES} CAPTCHA images!")
    print(f"Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
