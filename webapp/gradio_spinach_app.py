"""Spinach Leaf Classifier — Gradio App"""
import os
from pathlib import Path

import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models


def resolve_model_path() -> Path:
    env = os.environ.get("MODEL_PATH")
    if env and Path(env).exists():
        return Path(env)
    relative = Path(__file__).parent.parent / "models" / "mobilenet_best.pth"
    if relative.exists():
        return relative
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(
            repo_id="hemanthkumar0406/spinach-leaf-mobilenetv2",
            filename="mobilenet_best.pth",
        )
        return Path(p)
    except Exception as e:
        raise FileNotFoundError(f"Model not found: {e}")


def build_model(path: Path) -> torch.nn.Module:
    model = models.mobilenet_v2(weights=None)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.2),
        torch.nn.Linear(1280, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(256, 1),
    )
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


MODEL = build_model(resolve_model_path())

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

CLASS_NAMES = {0: "Big Leaf 🌿", 1: "Small Leaf 🌱"}


def predict(image: Image.Image):
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        logit = MODEL(tensor)[0, 0].item()
    prob = torch.sigmoid(torch.tensor(logit)).item()
    big_conf   = round((1 - prob) * 100, 1)
    small_conf = round(prob * 100, 1)
    label = CLASS_NAMES[1 if prob >= 0.5 else 0]
    return label, {"Big Leaf": big_conf, "Small Leaf": small_conf}


app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Spinach Leaf Image"),
    outputs=[
        gr.Text(label="Prediction"),
        gr.Label(label="Confidence"),
    ],
    title="Spinach Leaf Classifier 🌿",
    description="Upload a spinach leaf image to classify it as Big or Small.",
)

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=True,
    )