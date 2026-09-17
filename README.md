# 👁️‍🗨️ Day 04: Specialized Architectures & Transformers

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet18-ee4c2c)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-DistilBERT-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

A dual-domain deep learning repository demonstrating Transfer Learning for Computer Vision (using pre-trained ResNet18) and fine-tuning Transformer models for Natural Language Processing (using Hugging Face DistilBERT).

---

## 🛠️ Key Features

* **Computer Vision (CV):** Fine-tuned `ResNet18` feature extractor with modified classifier heads for custom image classification tasks.
* **Natural Language Processing (NLP):** Utilized `DistilBERT` Tokenizer and Transformer models for sequence classification and sentiment scoring.
* **Hugging Face Ecosystem:** Dynamic tokenization, padding, truncation, and artifact exportation using `transformers`.

---

## 📂 Repository Structure

```text
Day04_Specialized_Architectures_Transformers/
├── cv_transfer_learning.py    # ResNet18 image classification pipeline
├── nlp_transformer.py         # DistilBERT text classification pipeline
├── .gitignore
└── README.md

🚀 How to Run
1. Setup Environment
Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install torch torchvision transformers datasets scikit-learn pandas pillow matplotlib
2. Execute Computer Vision Pipeline
Bash
python cv_transfer_learning.py
3. Execute NLP Transformer Pipeline
Bash
python nlp_transformer.py
Part of the 7-Day Machine Learning Engineering Challenge.
