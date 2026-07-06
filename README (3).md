# Facial Emotion Recognition — FER2013 (EfficientNetV2M + TTA)

A deep-learning facial emotion classifier trained on the **FER2013** dataset, extending a
DCNN baseline paper (65.68% validation accuracy) with a stronger backbone, better
regularization, and Test-Time Augmentation — reaching **68.42% TTA test accuracy**.

Built as an ICMLDE 2023-based research extension project.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Model Architecture](#model-architecture)
- [Key Techniques](#key-techniques)
- [Training Strategy](#training-strategy)
- [Results](#results)
- [Base Paper vs. This Implementation](#base-paper-vs-this-implementation)
- [Tech Stack](#tech-stack)
- [Future Work](#future-work)
- [License](#license)

---

## Overview

Facial Emotion Recognition (FER) classifies a face image into one of 7 emotions: **Angry,
Disgust, Fear, Happy, Sad, Surprise, Neutral**. This project reproduces and extends a base
DCNN research paper's approach, replacing the custom 14-layer CNN with a pretrained
**EfficientNetV2M** backbone and modern training tricks (MixUp, Focal Loss, warm-restart
LR scheduling, and 10-crop TTA) to beat the paper's reported validation accuracy.

## Dataset

**[FER2013](https://www.kaggle.com/datasets/msambare/fer2013)** — 35,887 grayscale face
images (48×48), labeled with 7 emotion classes, split roughly 28,709 train / 7,178
test. The dataset is **not included** in this repo — download it from Kaggle and either:
- Place it as pre-split image folders (`train/<emotion>/*.png`, `test/<emotion>/*.png`), or
- Provide the raw FER2013 CSV (the notebook auto-converts CSV rows into images)

The notebook auto-detects which format is present.

## Project Structure

```
fer2013-emotion-recognition/
├── FER2013_Max_Accuracy.ipynb      # main notebook (annotated, section-by-section)
├── docs/
│   └── FER_Presentation.pptx       # project write-up slides (architecture, results)
├── data/                            # (not tracked) place FER2013 data here
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/fer2013-emotion-recognition.git
cd fer2013-emotion-recognition

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add the dataset
mkdir -p data
# download FER2013 from Kaggle into data/ (as folders or CSV)

# 5. Launch the notebook
jupyter notebook FER2013_Max_Accuracy.ipynb
```

> **GPU strongly recommended.** This model (EfficientNetV2M, ~54M params, 3 training
> phases) was designed for a Kaggle T4/P100 GPU environment; CPU training will be
> extremely slow.

## Model Architecture

- **Backbone:** EfficientNetV2M, ImageNet-pretrained (~54M parameters), input 128×128×3
- **Head:** Dual pooling (GlobalAvgPool + GlobalMaxPool) → BatchNorm → Dropout →
  Dense(768, Swish) → Dense(384, Swish) → Dense(7, softmax)
- **Loss:** Focal Loss (γ=2.0, label smoothing=0.1) to handle class imbalance
  (Disgust is under-represented)
- **Optimizer:** AdamW with weight decay 1e-4

## Key Techniques

| Technique | Purpose |
|---|---|
| **MixUp augmentation** (α=0.3) | Blends pairs of training images/labels as a regularizer; ~2–3% accuracy boost |
| **Focal Loss** | Down-weights easy examples, focuses training on hard/misclassified ones |
| **Flat-then-cosine LR schedule** | Linear warmup → flat plateau → cosine decay, avoiding the too-fast decay of pure cosine scheduling |
| **Class weights** | Balances the small `Disgust` class during training |
| **10-crop TTA** | 5 crops (center + 4 corners) × horizontal flip, averaged at inference for the final accuracy boost |

## Training Strategy

Three-phase progressive fine-tuning:

| Phase | What's Trained | Epochs | LR |
|---|---|---|---|
| **1 — Head only** | Classification head, backbone frozen | up to 20 | 1e-3 |
| **2 — Partial fine-tune** | Top 200 backbone layers unfrozen | up to 60 | 2e-4 |
| **3 — Full fine-tune** | Entire backbone unfrozen, lighter augmentation | up to 40 | 5e-6 |

Each phase loads the **best** checkpoint (by validation accuracy) from the previous
phase — not just the final epoch's weights — to avoid carrying forward an overfit state.

## Results

| Model | Val Accuracy | F1 |
|---|---|---|
| DCNN (base paper) | 65.68% | 0.6358 |
| EfficientNet (baseline comparison) | 58.41% | 0.28 |
| ResNet50 (baseline comparison) | 54.67% | 0.20 |
| VGGNet (baseline comparison) | 51.11% | 0.15 |
| **This model — raw test** | 64.49% | — |
| **This model — with 10-crop TTA** | **68.42%** | — |

**+2.74%** over the base paper's 65.68% validation accuracy.

## Base Paper vs. This Implementation

| | Base Paper (DCNN) | This Implementation |
|---|---|---|
| Architecture | Custom 14-layer DCNN | EfficientNetV2M + custom head |
| Parameters | ~2.4M | ~54M (backbone) |
| Image size | 48×48 grayscale | 128×128 RGB |
| Training | 100 epochs, single phase | 3 phases (20+60+40 epochs) |
| Loss | Categorical cross-entropy | Focal Loss (γ=2, label smoothing=0.1) |
| Optimizer | Adam (lr=0.001) | AdamW (weight decay=1e-4) |
| Augmentation | Rotation, flip, shear, zoom | + MixUp (α=0.3), brightness |
| Evaluation | Simple forward pass | 10-crop TTA (5-crop × flip) |
| Train acc. | 82.56% | 64.49% (raw test) |
| Val/Test acc. | 65.68% | 68.42% (TTA) |

## Tech Stack

- Python, TensorFlow / Keras
- EfficientNetV2M (`tf.keras.applications`)
- NumPy, Pandas, Pillow
- scikit-learn (metrics, class weighting)
- Matplotlib, Seaborn (visualization)

## Future Work

- Ensemble multiple backbones for further accuracy gains
- Real-time webcam deployment
- Multi-modal emotion recognition (face + voice)
- Training on larger / more balanced emotion datasets

## License

This project is licensed under the MIT License.
