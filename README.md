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

# 🌿 Spinach Leaf Classification

A deep learning project to classify spinach leaves as **Big** or **Small** using MobileNetV2.

Built as part of research at **IIITDM Kurnool**.

---

## 🚀 Live Demo

> Run the app locally and open: **http://127.0.0.1:7860**

---

## 🧠 Model

| Property | Details |
|----------|---------|
| Architecture | MobileNetV2 |
| Task | Binary Classification |
| Input Size | 224 × 224 |
| Output | Big Leaf / Small Leaf |
| Loss Function | BCEWithLogitsLoss |

---

## 🏷️ Classes

| Label | Index |
|-------|-------|
| Big Leaf 🌿 | 0 |
| Small Leaf 🌱 | 1 |

---

## 📊 Dataset

| Split | Description |
|-------|-------------|
| Train | Augmented images |
| Validation | Cropped images |
| Test (Big) | Real field images — large leaves |
| Test (Small) | Real field images — small leaves |

**[📥 Download Dataset – Google Drive](https://drive.google.com/drive/folders/1prjz5-J3Nbj9h4TXb427Xe83CRALaQlN?usp=drive_link)**

---

## 🗂️ Project Structure