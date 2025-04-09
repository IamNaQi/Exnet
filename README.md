# 🧠 EXNet: An Improved U-Net Architecture for Accurate Sperm Segmentation

This repository contains the official implementation of **EXNet**, a deep learning model proposed in Chapter 3 of the thesis:
> 📘 _Automated Deep Learning based Sperm Segmentation and Morphological Analysis for Enhancing Male Infertility Diagnosis_  
> Author: [Your Name]  
> University of Science and Technology of China (USTC)

---

## 📌 Overview

**EXNet** is a customized and improved U-Net architecture tailored for binary sperm segmentation from H&E-stained microscopic images. It introduces:

- 🧱 Dual Convolutional Blocks  
- 🧲 Multi-scale Attention Mechanisms  
- 🌐 Spatial Feature Extractor (Extractor Module)  
- 🎯 Goal: Enhance segmentation precision for sperm cells under varying imaging conditions.

---

## 📂 Directory Structure

```bash
exnet-sperm-segmentation/
│
├── dataset/
│   ├── train/
│   ├── test/
│   └── masks/
│
├── models/
│   └── exnet.py              # EXNet architecture
│
├── utils/
│   ├── dataloader.py         # Custom dataset class and transforms
│   └── metrics.py            # Dice, IoU, Precision, Recall, etc.
│
├── train.py                  # Training loop
├── test.py                   # Evaluation script
├── visualize.py              # Visualization of predictions
├── requirements.txt
└── README.md
