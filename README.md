# 🌿 Spinach Leaf Classification

A deep learning project to classify spinach leaves into **Big Leaf** or **Small Leaf** categories using MobileNetV2 transfer learning.

---

## 📌 Project Overview

| Item | Details |
|---|---|
| Task | Binary Image Classification |
| Classes | Big Leaf, Small Leaf |
| Best Model | MobileNetV2 (Transfer Learning) |
| Training Accuracy | ~99.80% |
| Validation Accuracy | ~86.67% |
| Interface | Gradio Web App |

---

## 📁 Project Structure

```
Spinach_Leaf_Classification/
│
├── dataset/                    # Raw spinach leaf images
│
├── models/                     # Trained model checkpoints
│   ├── mobilenet_best.pth      # MobileNetV2 weights
│   ├── custom_cnn_best.pth     # Custom CNN weights
│   └── xgboost_model.json      # XGBoost model
│
├── notebook/                   # Jupyter notebooks
│   ├── mobilenet/
│   │   └── mobilenet_spinach_leaf_classification.ipynb
│   ├── custom_cnn/
│   │   └── custom_CNN_spinach_leaf_classification.ipynb
│   └── xgboost/
│       └── xgboost_spinach_leaf_classification.ipynb
│
├── results/                    # Training results and plots
│   ├── class_indices.csv
│   ├── mobilenet/
│   │   ├── history.json
│   │   ├── confusion_matrix.png
│   │   └── training_curves.png
│   ├── custom_cnn/
│   │   ├── history.json
│   │   ├── confusion_matrix.png
│   │   └── training_curves.png
│   └── xgboost/
│       ├── results_summary.csv
│       ├── confusion_matrix.png
│       └── feature_importance.png
│
├── webapp/
│   └── gradio_spinach_app.py   # Gradio web application
│
├── requirements.txt
└── README.md
```

---

## 🧠 Models Developed

### 1. MobileNetV2 (Best Model)
- Transfer learning with ImageNet pretrained weights
- Input size: 224 × 224
- Custom classifier head
- Training: 30 epochs
- Test Accuracy: ~99.80%

### 2. Custom CNN
- Built from scratch
- Test Accuracy: 95.56%

### 3. XGBoost
- Traditional ML approach
- Test Accuracy: 97.78%

---

## 📊 Dataset

| Split | Big | Small | Total |
|---|---|---|---|
| Training | 255 | 255 | 510 |
| Validation | 22 | 23 | 45 |
| Testing | 23 | 22 | 45 |
| **Total** | **300** | **300** | **600** |

- Dataset is **balanced** (equal Big and Small images)
- Data augmentation applied to training set only

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10.11
- CUDA compatible GPU (recommended)

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Spinach_Leaf_Classification.git
cd Spinach_Leaf_Classification
```

### Step 2 — Create virtual environment
```bash
python -m venv spinach_env
spinach_env\Scripts\activate        # Windows
source spinach_env/bin/activate     # Mac/Linux
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Install PyTorch with CUDA
```bash
pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

---

## 🚀 Run the Gradio App

```bash
cd webapp
python gradio_spinach_app.py
```

Then open your browser and go to:
```
http://localhost:7860
```

---

## 🖼️ How to Use

1. Open the Gradio app in your browser
2. Upload a spinach leaf image
3. Click **Predict**
4. View the result — **Big Leaf** or **Small Leaf** with confidence score

---

## 📈 Class Mapping

| Class | Index | Sigmoid Output |
|---|---|---|
| Big Leaf | 0 | < 0.5 |
| Small Leaf | 1 | ≥ 0.5 |

---

## 🛠️ Tech Stack

- **Python** 3.10.11
- **PyTorch** 2.5.1
- **Torchvision** 0.20.1
- **Gradio** 6.32.0
- **XGBoost** 2.1.2
- **scikit-learn** 1.5.2
- **Pillow**, **NumPy**, **Matplotlib**

---

## 📝 Notes

- Model checkpoint `mobilenet_best.pth` must be present in `models/` folder
- Do not modify dataset folder structure
- Validation accuracy is ~86.67% due to limited dataset size
- For better accuracy, consider collecting more diverse spinach images

---

## 👤 Author

**Hemanth Kumar**
- Interview Project — Spinach Leaf Classification
- Built with PyTorch and Gradio
