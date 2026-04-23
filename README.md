# Multimodal Property Valuation Using Deep Learning

**Author:** Dorcas Ibrahim  
**Institution:** Fisk University  
**Course:** Machine Learning — Spring 2026

---

## Overview

This project proposes a multimodal deep learning approach to automated property valuation that combines both structured housing data and listing photographs. By fusing a Vision Transformer (ViT-B/16) image branch with a tabular neural network branch, the model learns to value properties using both numerical features and visual information.

The central hypothesis is that a multimodal model will outperform single-modality baselines, demonstrating the measurable value of visual information in automated valuation.

---

## Repository Structure
multimodal-property-valuation/
│
├── 01_tabular_baseline.ipynb        # MLP on Ames Housing structured data
├── 02_image_baseline.ipynb          # ViT-B/16 fine-tuned on listing images
├── 03_multimodal_fusion.ipynb       # Late fusion multimodal model
├── 04_evaluation_viz.ipynb          # GradCAM, attention maps, results comparison
└── README.md

---

## Datasets

### 1. Ames Housing Dataset
- **Source:** Kaggle — [Ames Housing Dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)
- **Used for:** Tabular-only baseline
- 2,930 residential property records with 79 structured features and sale prices

### 2. Houses Dataset (Ahmed & Moustafa, 2016)
- **Source:** [GitHub — emanhamed/Houses-dataset](https://github.com/emanhamed/Houses-dataset)
- **Used for:** Image-only baseline and multimodal fusion model
- 535 houses, each with 4 listing images (frontal, bedroom, bathroom, kitchen) and structured metadata (bedrooms, bathrooms, area, zipcode, price)
- Citation: H. Ahmed E. and Moustafa M. (2016). *House Price Estimation from Visual and Textual Features.* IJCCI 2016.

---

## Models

### 1. Tabular-Only MLP (Baseline)
- 4-layer fully connected network with BatchNorm and Dropout
- Input: 79 structured features from Ames Housing Dataset
- Trained for 200 epochs with Adam optimizer

### 2. Image-Only ViT-B/16 (Baseline)
- Pretrained Vision Transformer (ViT-B/16) fine-tuned on listing images
- 4 images per house averaged into a single tensor
- Differential learning rates: 1e-5 for backbone, 1e-4 for regression head

### 3. Multimodal Fusion Model
- **Late fusion** architecture combining ViT-B/16 image branch + tabular MLP branch
- Both branches projected to 256-dimensional embeddings, concatenated, and passed through a joint regression head
- Trained end-to-end with differential learning rates

---

## Results

| Model | Dataset | RMSE | MAE | R² |
|-------|---------|------|-----|----|
| Tabular-only MLP | Ames Housing | $66,129 | $44,145 | 0.393 |
| Image-only ViT-B/16 | Houses Dataset | $317,932 | $223,730 | 0.154 |
| Multimodal Fusion | Houses Dataset | $267,434 | $201,047 | 0.401 |

> Note: The tabular baseline uses a different dataset than the image and multimodal models. Direct cross-dataset metric comparison should be interpreted with this distinction in mind. The primary controlled comparison is between the image-only and multimodal models, both trained on identical data splits of the Houses Dataset.

---

## Setup & Reproduction

All notebooks are designed to run on **Google Colab** with a T4 GPU.

### Requirements
```bash
pip install torch torchvision timm scikit-learn pandas numpy matplotlib seaborn
```

### Running the notebooks
1. Open each notebook in Google Colab
2. Set runtime to **T4 GPU** (Runtime → Change runtime type)
3. Run cells sequentially

---

## Visualizations

- EDA plots (price distributions, feature correlations)
- Training/validation loss curves for all 3 models
- GradCAM attention maps showing which image regions influence price predictions
- Final comparison bar chart across all models

---

## Future Work

- Incorporate larger paired image + price datasets (e.g., Zillow listings)
- Explore early fusion strategies as an alternative to late fusion
- Apply data augmentation to improve image model generalization
- Extend to multi-city generalization beyond Southern California and Arizona

---

## Acknowledgements

- Ames Housing Dataset: De Cock, D. (2011). *Ames, Iowa: Alternative to the Boston Housing Data.* Journal of Statistics Education, 19(3).
- Houses Dataset: Ahmed & Moustafa (2016). *House Price Estimation from Visual and Textual Features.* IJCCI 2016.
