"""
Gradio app for testing CAPTCHA model.
Allows uploading CAPTCHA images and getting predictions with preprocessing.
"""
import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import string
from pathlib import Path
import numpy as np
import cv2

from src.model import CTCCaptchaModel


# Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHARACTERS = string.digits + string.ascii_lowercase + string.ascii_uppercase
MODEL_PATH = Path("models/captcha_model_v4.pth")

# Load model
model = CTCCaptchaModel(num_classes=len(CHARACTERS), use_attention=True)

# Load checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

# Image preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((60, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


def preprocess_image(image):
    """
    Preprocess image: grayscale, denoising, and thresholding.
    
    Args:
        image: PIL Image
    
    Returns:
        Preprocessed PIL Image
    """
    # Convert to grayscale numpy array
    img_array = np.array(image.convert('L'))

    # If background is dark (mean < 127), invert so we get dark text on light background
    if img_array.mean() < 127:
        img_array = 255 - img_array

    # Apply Otsu's thresholding
    _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological closing to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Convert back to PIL Image
    return Image.fromarray(processed)


def predict_captcha(image, ground_truth=""):
    """
    Predict CAPTCHA text from image with preprocessing.
    
    Args:
        image: PIL Image or numpy array
        ground_truth: Optional ground truth text for comparison
    
    Returns:
        Tuple of (prediction result, preprocessed image)
    """
    try:
        # Convert to PIL Image if numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Resize image if not standard dimensions (60x160)
        if image.size != (160, 60):
            image = image.resize((160, 60), Image.LANCZOS)
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Convert to tensor and predict
        image_tensor = transform(processed_image).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            decoded = model.predict(image_tensor)

        # Decode first (and only) batch element
        pred_indices = decoded[0] if decoded else []
        predicted_text = ''.join([
            CHARACTERS[idx] for idx in pred_indices 
            if 0 <= idx < len(CHARACTERS)
        ])
        
        # Format output with styling
        result = f"### 🎯 Prediction Result\n\n"
        result += f"# **{predicted_text}**\n\n"
        result += f"*Length: {len(predicted_text)} characters*\n\n"
        
        if ground_truth.strip():
            ground_truth = ground_truth  # Keep case sensitive
            is_correct = predicted_text == ground_truth
            result += f"**Expected:** {ground_truth}\n\n"
            if is_correct:
                result += "## ✅ **CORRECT!**"
            else:
                result += f"## ❌ **INCORRECT**"
        
        return result, processed_image
    
    except Exception as e:
        return f"❌ **Error:** {str(e)}", None


def extract_from_filename(filename):
    """Extract text from CAPTCHA filename (format: TEXT_INDEX.png)."""
    if filename and hasattr(filename, 'name'):
        stem = Path(filename.name).stem
        text = stem.split('_')[0]
        return text
    return ""


# Create Gradio interface
with gr.Blocks(title="🔐 CAPTCHA Breaker", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    <div style="text-align: center; padding: 20px;">
    
    # 🔐 CAPTCHA Breaker
    
    ### Advanced AI-Powered CAPTCHA Recognition
    
    Powered by **CNN + LSTM + Self-Attention** neural network
    
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("#### 📸 Upload Your CAPTCHA")
            image_input = gr.Image(
                type="pil",
                label="Drop CAPTCHA image here",
                image_mode="L"
            )
            
            with gr.Row():
                ground_truth_input = gr.Textbox(
                    label="Expected Answer (optional)",
                    placeholder="Type here to verify accuracy",
                    lines=1,
                    scale=3
                )
                predict_button = gr.Button(
                    "🔍 Decode",
                    variant="primary",
                    scale=1
                )
        
        with gr.Column(scale=2):
            gr.Markdown("#### 🎯 Results")
            output = gr.Markdown(
                "<div style='text-align: center; padding: 40px; color: #888;'>Upload an image to get started</div>"
            )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### 🔬 Preprocessing Steps Applied:")
            gr.Markdown("""
            - ✓ Auto-resize to 60×160 (if needed)
            - ✓ Grayscale conversion
            - ✓ Otsu's thresholding
            - ✓ Morphological closing (denoising)
            - ✓ Tensor normalization
            - ✓ Variable length support (3-7 chars)
            - ✓ Lowercase + Uppercase + Digits
            """)
        
        with gr.Column():
            gr.Markdown("#### 📊 Character Set:")
            gr.Markdown("""
            - **Digits:** 0-9
            - **Lowercase:** a-z
            - **Uppercase:** A-Z
            - **Total:** 62 characters
            """)
        
        with gr.Column():
            gr.Markdown("#### 🖼️ Processed Image:")
            preprocessed_image = gr.Image(
                label="Input After Preprocessing",
                type="pil"
            )
    
    # Info section
    with gr.Accordion("ℹ️ Model Architecture & Performance", open=False):
        gr.Markdown("""
        ### 🏗️ Architecture
        
        ```
        Input Image (1, 60, 160) [Auto-resized if needed]
            ↓
        CNN: 4 Convolutional Blocks
          • Progressive feature extraction
          • 1→32→64→128→256 channels
            ↓
        Bidirectional LSTM: 2 layers
          • 256 hidden units each direction
          • Learns sequential dependencies
            ↓
        Self-Attention: 4 heads
          • Refines character representations
          • Improves focus on important features
            ↓
        CTC Loss: Automatic Alignment
          • No bounding boxes needed!
          • Learns character positions automatically
            ↓
        Output: Variable-length prediction (3-7 characters)
        ```
        
        ### 📈 Model Capabilities (v3)
        
        | Feature | Details |
        |---------|---------|
        | **Model Version** | v3 (Latest) |
        | **Text Length** | 3-7 characters (variable) |
        | **Character Set** | 0-9, a-z, A-Z (62 total) |
        | **Architecture** | CNN + LSTM + Attention |
        | **Training Data** | 10,000 synthetic CAPTCHAs |
        | **Image Resize** | Automatic (any size → 60×160) |
        
        ### ⚠️ Known Limitations
        
        - 0 vs O confusion (visual similarity)
        - i vs l vs 1 confusion (very similar shapes)
        - Limited performance on decorative/stylized fonts
        - Sensitive to extreme image distortions
        """)
    
    # Connect buttons to prediction function
    predict_button.click(
        fn=predict_captcha,
        inputs=[image_input, ground_truth_input],
        outputs=[output, preprocessed_image]
    )
    
    # Auto-predict on image upload
    image_input.change(
        fn=lambda img: predict_captcha(img, ""),
        inputs=image_input,
        outputs=[output, preprocessed_image]
    )
    
    # Footer
    gr.Markdown("""
    ---
    <div style="text-align: center; color: #999; padding: 20px;">
    Built with PyTorch | Device: {device} | GitHub: vedchamp07/captcha-breaker
    </div>
    """.format(device=DEVICE))


if __name__ == "__main__":
    demo.launch(share=True)
