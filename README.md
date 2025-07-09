# AI Generated Text Detection Using Adversarial Learning

This repository contains the implementation of our CS577 project, where we explore improving the robustness of AI-generated text detection by using adversarial learning, inspired by the RADAR framework.

Original paper reference: [RADAR: Robust AI-text detection via adversarial learning](https://dl.acm.org/doi/10.5555/3666122.3666784)

---

## 🚀 Overview

Recent advancements in large language models (LLMs) have made it increasingly difficult to distinguish between machine-generated and human-written text. This project aims to:

* Investigate the robustness of AI-text detectors under adversarial paraphrasing.
* Develop a hybrid paraphrasing strategy combining backtranslation and neural paraphrasing.
* Train a distilBERT-based detector to classify texts as AI-generated or human-written.

---

## 📌 Features

* **Hybrid paraphrasing pipeline**: Combines multilingual backtranslation and lexical neural paraphrasing.
* **Adversarial training loop**: Reinforcement learning-based paraphraser competes against a binary classifier.
* **Evaluation on real-world data**: Includes a manually annotated dataset of LinkedIn posts.

---

## ⚙ Requirements

* Python >= 3.9
* PyTorch >= 2.6.0 with CUDA support
* HuggingFace `transformers` & `datasets`
* NLTK
* Helsinki-NLP models

---

## 🧰 Datasets

| Split      | Source                 | Count |
| ---------- | ---------------------- | ----- |
| Training   | OpenWebText (filtered) | 9,000 |
| Validation | OpenWebText (filtered) | 1,000 |
| Test       | LinkedIn posts         | 45    |

---

## 🚀 Implementation Details

### Hybrid Paraphrasing Pipeline

* **Backtranslation:** English → French → English using Helsinki-NLP.
* **Neural paraphrasing:** NLTK-based paraphraser to create more natural variations.

### Model Architecture

* **Detector:** `distilbert-base-uncased` binary classifier.
* **Paraphraser:** `t5-small` fine-tuned with PPO.

### Adversarial Training

* Paraphraser generates samples to fool the detector.
* Detector learns from these new samples to improve classification.

### Evaluation Metrics

* AUROC, Accuracy, F1, Precision, Recall.

---

## 🛠 Usage

```bash
git clone https://github.com/Subangkar/cs577-project.git
cd cs577-project
pip install -r requirements.txt
```

Run training:

```bash
python radar.py
```

Run evaluation:

```bash
python radar_evaluate.py
```

---

