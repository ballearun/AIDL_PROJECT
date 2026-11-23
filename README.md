# AIDL_PROJECT# 
🔥 Fire and Smoke Detection Using Deep Learning

Surveillance-grade, multi-class hazard recognition (**Fire / Smoke / Neutral**) with **Grad-CAM explainability**, benchmarked across **28 trained pipelines** (ResNet vs MobileNet families).

**B.Tech CSE – Deep Learning Course Project** | SR University, Warangal, Telangana, India

| Student | Roll Number | Department | Academic Year |
|---|---|---|---|
| B. Arun Kumar | 2403A510A0 | Computer Science & Engineering | 2026 |

---

## 📌 Highlights

| Metric (Test Set) | Value |
|---|---|
| Test accuracy | **96.96%** (414 / 427) |
| Weighted F1-score | **0.9696** |
| Weighted precision / recall | 0.9697 / 0.9696 |
| Multi-class ROC-AUC | **0.9972** |
| Inference latency | **1.46 ms / image** |
| Parameters | **3.00 M** |
| Champion model | **MobileNetV3Large + RMSprop** (best epoch 7 / 50) |

- Only **13 misclassifications** out of 427 unseen images.
- **Zero** true fire images were classified as Neutral; ambiguous cases fall back to Smoke, so an alarm is still raised.
- The champion is **2.7× faster** and **11.7× lighter** than the runner-up ResNet101 (F1 0.9672, 5.17 ms, 42.66 M parameters).

---

## 🎯 Problem Statement

Traditional physical fire detectors (ionization/photoelectric) suffer from severe transport delays in high-ceiling halls, warehouses and open spaces, and give no spatial localization of the fire origin. This project classifies surveillance / drone camera frames as **Fire**, **Smoke** or **Neutral** so that localized alarms can be triggered before flame propagation, while ambient conditions (sunlight reflections, street lamps, red clothing, moving shadows) do **not** trigger false alarms.

---

## 🧠 Objectives

1. **Benchmark dual model families** – 7 backbones: ResNet50, ResNet101, ResNet50V2, ResNet101V2, MobileNetV1, MobileNetV2, MobileNetV3Large.
2. **Optimizer & convergence analysis** – Adam, AdamW, RMSprop, SGD over 50 epochs (7 × 4 = **28 pipelines**).
3. **Feature caching & data augmentation** – 6-way geometric/photometric augmentation (+5,970 samples) with frozen ImageNet transfer learning.
4. **Rigorous held-out evaluation** – 427 unseen images using Accuracy, Precision, Recall, F1-score and ROC-AUC.
5. **Visual explainability** – Grad-CAM heatmaps for safety auditing.

---

## 🗂️ Dataset

**DeepQuestAI Fire & Smoke Dataset** (GitHub benchmark repository): real-world indoor and outdoor hazard scenarios.

| Class | Images | Share |
|---|---|---|
| 0 – Fire | 1,000 | 35.2% |
| 1 – Neutral | 844 | 29.7% |
| 2 – Smoke | 1,000 | 35.2% |
| **Total (valid, curated)** | **2,844** | 100% |

- 156 corrupt / unreadable files were removed during cleaning.
- **Stratified 70 / 15 / 15 split with zero overlap:** Training 1,990 · Validation 427 · Testing 427 (150 Fire, 127 Neutral, 150 Smoke).
- **Augmentation (training split only):** 3 random copies per image (+5,970) → **7,960** training samples. Validation and test splits stay 100% unaugmented.
- Dataset source: <https://github.com/DeepQuestAI/Fire-Smoke-Dataset>

---

## 🏗️ Methodology

```
Input Stream  →  Preprocessing  →  Backbone CNN  →  Classifier Head  →  Grad-CAM  →  Decision Logic
(CCTV/Drone)     (224×224 RGB)     (frozen,          Dropout(0.3) +      (spatial     (confidence check
                                    ImageNet)         Dense(3, softmax)   attention)    & alarm dispatch)
```

- **Architecture comparison:** ResNet family (residual skip connections) vs MobileNet family (depthwise separable convolutions, inverted residuals, squeeze-and-excitation attention).
- **Feature caching:** pretrained ImageNet backbones are frozen; global-average-pooled feature vectors (1024-d / 2048-d) are extracted and cached in RAM once, cutting 50-epoch head training to ~22 s per pipeline.
- **Classifier head:** `Input(dim) → Dropout(0.3) → Dense(3, softmax)`.
- **Augmentation:** horizontal flip, rotation ±0.08 rad, zoom 15%, shift 10%, brightness ±20%, contrast ±20%.
- **Loss & LR:** Sparse Categorical Crossentropy; `ReduceLROnPlateau` (factor 0.5, patience 2); best validation-loss weights restored.

### Training configuration

| Setting | Value |
|---|---|
| Framework | TensorFlow 2.21 / Keras 3 (GPU) |
| Input size | 224 × 224 × 3 RGB |
| Head batch size | 64 |
| Optimizers | Adam, AdamW, RMSprop, SGD |
| Total runs | 28 (7 models × 4 optimizers) |
| Epochs | 50 per run |

---

## 🚨 Alarm Decision Logic

The neural model outputs class probabilities and Grad-CAM heatmaps; a separate rule-based safety engine runs the site-specific response.

| Verdict | Meaning | Typical response |
|---|---|---|
| 🔴 **FIRE DETECTED** | Active flames with high probability (≥ 50%) | Immediate audible/visual alarms, emergency dispatch, sprinkler suppression |
| 🟡 **SMOKE DETECTED** | Smoke plume in the incipient stage | Early warning, HVAC air-damper shutoff, security video inspection |
| 🟢 **NEUTRAL / SAFE** | Normal activity (sunlight, headlights, dust, motion) | Logged silently, no alarm |

---

## 📊 Results

### Family comparison

| Family / Model | Mean F1 | Latency | Params |
|---|---|---|---|
| MobileNet family | 0.9524 | 1.48 ms | 2.83 M |
| ResNet family | 0.9526 | 4.02 ms | 33.12 M |
| **Winner: MobileNetV3Large** | **0.9696** | **1.46 ms** | **3.00 M** |
| Runner-up: ResNet101 | 0.9672 | 5.17 ms | 42.66 M |

### Confusion matrix reading (champion model)

- **Fire (150):** 146 correct (97.33% recall), 0 → Neutral, 4 → Smoke.
- **Neutral (127):** 124 correct (97.64% recall, 99.20% precision), 1 → Fire.
- **Smoke (150):** 144 correct (96.00% recall), 5 → Fire, 1 → Neutral.

### Key insights

- **Squeeze-and-Excitation advantage:** MobileNetV3Large reaches the top F1 using channel attention that weights flame and smoke feature maps.
- **Diminishing returns for heavy models:** ResNet101 needs 14.2× more parameters and 3.5× slower inference for a slightly lower F1.
- **Edge feasibility:** the MobileNet family averages ~1.48 ms/image (~675 FPS), enabling real-time multi-camera CCTV processing on low-power hardware.
- **Optimizer hierarchy:** RMSprop and AdamW outperformed vanilla SGD across both families.
- **Explainability:** Grad-CAM shows attention on flame cores and rising smoke contours, with diffuse low activations for Neutral scenes.

> 📷 *Add your figures here (confusion matrices, ROC curves, training curves, Grad-CAM overlays, prediction grid), for example:*
> `![Grad-CAM](images/gradcam.jpg)`

---

## ⚠️ Limitations

- **Static 2D images:** no temporal flicker or smoke-rise dynamics from continuous video.
- **Atmospheric confusion:** dense fog or industrial steam can occasionally trigger false smoke warnings.
- **Low-light night contrast:** CCTV without infrared/thermal illumination degrades smoke visibility.
- **Single-split evaluation:** testing was done on a held-out benchmark split; multi-site field trials remain essential.

---

## 🚀 Future Scope

- **Spatio-temporal video models:** ConvLSTM / 3D-CNNs (Video Swin, SlowFast) for temporal flame flicker.
- **Edge deployment:** INT8 quantization via TensorRT / TFLite for Raspberry Pi 5 and Jetson Orin Nano.
- **Multi-spectral fusion:** combine RGB with Long-Wave Infrared (LWIR) thermal cameras for zero-light surveillance.
- **Autonomous UAV patrol:** lightweight MobileNetV3 on drone swarms for forest-fire and wildfire reconnaissance.

---

## ⚙️ Getting Started

> Replace the placeholders below with your actual repository details.

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# 2. Install dependencies (TensorFlow 2.21 / Keras 3)
pip install -r requirements.txt

# 3. Download the dataset
#    https://github.com/DeepQuestAI/Fire-Smoke-Dataset

# 4. Open and run the notebook
jupyter notebook <your-notebook-name>.ipynb
```

---

## 🛠️ Tech Stack

- TensorFlow 2.21 / Keras 3 (GPU)
- ImageNet-pretrained ResNet and MobileNet backbones
- Grad-CAM for explainability
- DeepQuestAI Fire & Smoke Dataset

---

## 📚 References

1. DeepQuestAI, "Fire and Smoke Dataset: A Benchmark Dataset for Real-World Fire and Smoke Detection in Surveillance Systems," GitHub Repository, 2019. <https://github.com/DeepQuestAI/Fire-Smoke-Dataset>
2. K. He, X. Zhang, S. Ren, J. Sun, "Deep Residual Learning for Image Recognition," CVPR, pp. 770–778, 2016.
3. A. Howard et al., "Searching for MobileNetV3," ICCV, pp. 1314–1324, 2019.
4. M. Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks," CVPR, pp. 4510–4520, 2018.
5. R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," ICCV, pp. 618–626, 2017.
6. M. Abadi et al., "TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems," tensorflow.org.

---

## 👤 Author

**B. Arun Kumar** – Roll No. 2403A510A0
B.Tech CSE, SR University

⭐ If you find this project useful, please give it a star!
