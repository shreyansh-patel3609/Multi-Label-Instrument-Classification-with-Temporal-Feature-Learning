# Multi-Label-Instrument-Classification-with-Temporal-Feature-Learning
# 🎵 Multi-Label Instrument Classification with Temporal Feature Learning

---

## 📌 Project Overview

Recognizing musical instruments in **polyphonic audio** (where multiple instruments play simultaneously) is a challenging task in **Music Information Retrieval (MIR)**.

Traditional baseline approaches for the **OpenMIC-2018 dataset** rely on **aggregated features** and **Random Forest models**, which ignore the **temporal structure** present in audio.

This project introduces a **Temporal 1-D Convolutional Neural Network (Conv1D)** to capture **time-varying audio patterns**, enabling improved **multi-label instrument classification** performance.

---

## ❓ Research Question

**Does incorporating temporal information using a 1-D convolutional model improve multi-instrument recognition performance compared to a non-temporal Random Forest baseline?**

---

## 📂 Dataset

**Dataset Used:**
**OpenMIC-2018**

**Key Properties:**

* ~20,000 short audio clips
* 20 instrument labels
* Multi-label annotations
* Precomputed **VGGish embeddings** used as input features

Input Data Files:

* `.npz` → VGGish feature embeddings
* `.csv` → Instrument relevance labels

---

## ⚙️ Methodology

The system follows a structured pipeline:

1. **Feature Loading**

   * Load VGGish embeddings from `.npz`
   * Preserve temporal frames for Conv1D model
   * Aggregate frames for Random Forest baseline

2. **Label Processing**

   * Convert long-format CSV labels into binary multi-label format
   * Apply relevance thresholding

3. **Baseline Model**

   * Aggregated VGGish embeddings
   * **Random Forest (One-vs-Rest)** classifier

4. **Temporal Model**

   * 1-D Convolution across time frames
   * Captures temporal dynamics in audio signals

5. **Evaluation Metrics**

   * Micro F1 Score
   * Macro F1 Score
   * Macro AUC

---

## 🧠 Model Architectures

### 🔹 Baseline Model — Random Forest

* Input: Averaged feature embeddings
* Classifier: One-vs-Rest Random Forest
* Fast training
* No temporal awareness

---

### 🔹 Proposed Model — Temporal Conv1D

Architecture:

* Conv1D Layers
* Batch Normalization
* Max Pooling
* Dropout Regularization
* Global Average Pooling
* Dense Layers
* Sigmoid Output Layer (multi-label classification)

Key Features:

* Preserves **temporal structure**
* Learns **time-dependent patterns**
* Improves classification accuracy

---

## 🚀 Usage

### 1️⃣ Random Forest Baseline

```bash
python openmic_temporal_conv1d.py \
  --features openmic-2018.npz \
  --labels openmic-2018-aggregated-labels.csv \
  --out rf_model.joblib \
  --model-type rf
```

---

### 2️⃣ Temporal Conv1D Model (PyTorch Recommended)

```bash
python openmic_temporal_conv1d.py \
  --features openmic-2018.npz \
  --labels openmic-2018-aggregated-labels.csv \
  --out temporal_conv1d_model.pt \
  --model-type conv1d \
  --backend pytorch
```

---

### Optional Parameters

```bash
--epochs 50
--batch_size 32
--learning_rate 0.001
--test_size 0.2
--relevance-threshold 0.5
```

---

## 📊 Results

| Model           | Training Time (sec) | Micro F1 | Macro F1 | Macro AUC |
| --------------- | ------------------- | -------- | -------- | --------- |
| Random Forest   | 51.38               | 0.54     | 0.47     | 0.95      |
| Temporal Conv1D | 156.30              | **0.70** | **0.65** | **0.97**  |

### 🔍 Key Findings

* Temporal Conv1D improved performance by **~39%**
* Temporal modeling significantly enhances **multi-instrument recognition**
* Conv1D captures sequential audio dependencies effectively

---

## 📦 Output Files

After training, the system generates:

* Trained Model

  * `.joblib` → Random Forest
  * `.pt` → PyTorch Conv1D
  * `.h5` → Keras Conv1D

* `predicted_labels.csv` → Binary predictions

* `predicted_scores.csv` → Prediction probabilities

* `*_history.csv` → Training history

---

## 🧰 Requirements

Install dependencies:

```bash
pip install numpy pandas scikit-learn joblib torch
```

Optional (for Keras backend):

```bash
pip install tensorflow
```

---

## 📈 Evaluation Metrics

The following metrics are used:

* **Micro F1 Score**

  * Measures overall classification accuracy

* **Macro F1 Score**

  * Treats all instruments equally

* **Macro AUC**

  * Measures classification separability

---

## 📚 Reference

Humphrey, E., Durand, S., & McFee, B. (2018).
**"OpenMIC-2018: An Open Dataset for Multiple Instrument Recognition."**
Proceedings of ISMIR 2018.

---

## 🎯 Conclusion

This project demonstrates that incorporating **temporal feature learning** through **1-D Convolutional Neural Networks** significantly improves multi-label musical instrument classification performance compared to traditional non-temporal models.

---

## 📬 Contact

**Shreyansh Rajeshkumar Patel**
📧 [patelsh1@b-tu.de](mailto:patelsh1@b-tu.de)
