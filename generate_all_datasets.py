"""
Generate CAPTCHA datasets for training, testing, and font testing.
Creates variable-length CAPTCHAs with lowercase and uppercase letters.
"""
from captcha.image import ImageCaptcha
import random
import string
from pathlib import Path

# Character set (digits + lowercase + uppercase letters)
CHARACTERS = string.digits + string.ascii_lowercase + string.ascii_uppercase

# Configuration
CONFIGS = {
    'train': {
        'dir': Path('data/train/raw'),
        'num_samples': 10000,
        'description': 'Training Dataset'
    },
    'test': {
        'dir': Path('data/test/raw'),
        'num_samples': 1000,
        'description': 'Test Dataset'
    },
    'font_test': {
        'dir': Path('data/font_test/raw'),
        'num_samples': 1000,
        'description': 'Font Test Dataset'
    }
}

MIN_LENGTH = 3  # Minimum captcha text length
MAX_LENGTH = 7  # Maximum captcha text length


def generate_captcha_text():
    """Generate random captcha text with variable length."""
    length = random.randint(MIN_LENGTH, MAX_LENGTH)
    return ''.join(random.choices(CHARACTERS, k=length))


def generate_dataset(dataset_type):
    """Generate a dataset (train/test/font_test)."""
    config = CONFIGS[dataset_type]
    output_dir = config['dir']
    num_samples = config['num_samples']
    description = config['description']
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize captcha generator
    generator = ImageCaptcha(width=160, height=60)
    
    print(f"\n{'='*60}")
    print(f"Generating {description}")
    print(f"{'='*60}")
    print(f"Samples: {num_samples}")
    print(f"Length range: {MIN_LENGTH}-{MAX_LENGTH} characters")
    print(f"Character set: {CHARACTERS}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    for i in range(num_samples):
        # Generate random text (variable length)
        captcha_text = generate_captcha_text()
        
        # Generate image
        image = generator.generate(captcha_text)
        
        # Save with filename as the captcha text
        filename = f"{captcha_text}_{i}.png"
        filepath = output_dir / filename
        
        generator.write(captcha_text, str(filepath))
        
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{num_samples} images...")
    
    print(f"\n✅ Successfully generated {num_samples} {description.lower()} images!")
    print(f"   Saved to: {output_dir}\n")


def main():
    """Generate all datasets."""
    print("\n" + "="*60)
    print("🔐 CAPTCHA Dataset Generation")
    print("="*60)
    
    for dataset_type in ['train', 'test', 'font_test']:
        generate_dataset(dataset_type)
    
    print("="*60)
    print("✨ All datasets generated successfully!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run: python preprocess.py")
    print("  2. Run: python train.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
