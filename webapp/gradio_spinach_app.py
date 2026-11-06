"""
Spinach Leaf Classifier — Gradio App
Model  : MobileNetV2
Classes: BIG LEAF / SMALL LEAF

Run:
    pip install gradio torch torchvision pillow
    python gradio_spinach_app.py

A public link (valid 72 hours) is printed in the terminal.
Anyone can open it on phone or PC — no installation needed.
"""

import torch
import torch.nn as nn
import gradio as gr

from pathlib import Path
from torchvision import models, transforms
from PIL import Image

# ── CONFIGURATION ────────────────────────────────────────────
MODEL_PATH = Path(r"D:\Spinach_Leaf_Classification\models\mobilenet_best.pth")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE   = 224

# ── BUILD MOBILENETV2 ARCHITECTURE ──────────────────────────
# Must match exactly what was used during training
def build_model():
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1)   # single logit → BCEWithLogitsLoss
    )
    return model

# ── LOAD SAVED WEIGHTS ───────────────────────────────────────
model = build_model()
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
)
model.to(DEVICE)
model.eval()
print("Model loaded. Device:", DEVICE)

# ── PREPROCESSING ────────────────────────────────────────────
# Same as val_test_tf used during training
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

# Class mapping — Big=0, Small=1 (alphabetical, same as ImageFolder)
IDX_TO_CLASS = {0: "BIG LEAF", 1: "SMALL LEAF"}

# ── PREDICTION FUNCTION ──────────────────────────────────────
# Gradio passes a PIL Image directly
def predict(pil_image):

    if pil_image is None:
        return "No image uploaded.", {}

    # Preprocess
    img    = pil_image.convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        logit      = model(tensor)
        prob_small = torch.sigmoid(logit).item()   # probability of SMALL LEAF

    prob_big = 1.0 - prob_small

    predicted_idx   = int(prob_small >= 0.5)
    predicted_class = IDX_TO_CLASS[predicted_idx]
    confidence      = prob_small if predicted_idx == 1 else prob_big

    # Gradio label output expects {"class_name": probability, ...}
    label_output = {
        "BIG LEAF":   round(prob_big,   4),
        "SMALL LEAF": round(prob_small, 4)
    }

    result_text = (
        f"Predicted  : {predicted_class}\n"
        f"Confidence : {confidence*100:.2f}%\n"
        f"Big Leaf   : {prob_big*100:.2f}%\n"
        f"Small Leaf : {prob_small*100:.2f}%"
    )

    return result_text, label_output


# ── GRADIO INTERFACE ─────────────────────────────────────────
with gr.Blocks(title="Spinach Leaf Classifier", theme=gr.themes.Default()) as app:

    # Header
    gr.Markdown(
        """
        # 🌿 Spinach Leaf Classifier
        **Model:** MobileNetV2 (Transfer Learning)
        **Task:** Classify spinach leaf as **BIG LEAF** or **SMALL LEAF**

        Upload any spinach leaf image — works on phone and PC.
        """
    )

    with gr.Row():

        # Left column — input
        with gr.Column():
            image_input = gr.Image(
                type="pil",
                label="Upload Spinach Leaf Image",
                height=300
            )
            classify_btn = gr.Button(
                "🔍 Classify Leaf",
                variant="primary",
                size="lg"
            )

        # Right column — output
        with gr.Column():
            text_output  = gr.Textbox(
                label="Prediction Result",
                lines=5,
                interactive=False
            )
            label_output = gr.Label(
                label="Class Probabilities",
                num_top_classes=2
            )

    # Examples section — optional, add your own leaf image paths
    gr.Examples(
        examples=[],           # add example image paths here if you have them
        inputs=image_input,
        label="Example Images"
    )

    gr.Markdown(
        """
        ---
        **How to use:**
        1. Click **Upload Spinach Leaf Image** or drag and drop
        2. Click **Classify Leaf**
        3. See prediction and confidence scores

        *Developed at IIITDM Kurnool — Applied AI Research*
        """
    )

    # Wire button to predict function
    classify_btn.click(
        fn      = predict,
        inputs  = [image_input],
        outputs = [text_output, label_output]
    )

    # Also trigger on image upload automatically
    image_input.change(
        fn      = predict,
        inputs  = [image_input],
        outputs = [text_output, label_output]
    )


# ── LAUNCH ───────────────────────────────────────────────────
if __name__ == "__main__":
    app.launch(
        share=True,       # ← generates public link anyone can open
        show_error=True
    )
    # share=True prints something like:
    # Running on public URL: https://xxxxxxxxxxxx.gradio.live
    # This link is valid for 72 hours.
    # Share it with anyone — works on mobile browser too.
