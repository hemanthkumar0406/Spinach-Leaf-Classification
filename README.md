---
title: Spinach Leaf Classification

emoji: 🌿
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# Spinach Leaf Classification

MobileNetV2-based binary classifier — Big leaf vs Small leaf.

## Classes
| Label | Index |
|-------|-------|
| Big   | 0     |
| Small | 1     |

## Run locally
```bash
pip install -r requirements.txt
python app.py
```

Then open browser: http://localhost:7860

## Run Gradio app directly
```bash
cd webapp
python gradio_spinach_app.py
```

## Dataset Download

The complete Spinach Leaf Classification dataset is available on Google Drive.

**[Download Dataset – Google Drive](https://drive.google.com/drive/folders/1prjz5-J3Nbj9h4TXb427Xe83CRALaQlN?usp=drive_link)**

### Dataset Classes
- Big Leaf
- Small Leaf

### Dataset Structure
- Training
- Validation
- Testing

The dataset is provided separately to avoid large files in the GitHub repository.
