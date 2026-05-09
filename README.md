# Indonesia Informal Settlement Classification

This project focuses on detecting and classifying informal settlements (slums) in Indonesia using high-resolution satellite imagery and machine learning techniques.

---

## Project Structure

```markdown
├── data/ # Datasets
├── notebooks/ # Exploratory analysis and experiments
├── models/ # Saved trained models
├── README.md
├── requirements.txt
└── .gitignore
```

## Dataset

This repository does not include the satellite imagery used for training and evaluation due to file size limitations.

To follow the notebook, place the labeled image tiles in the following directory structure:

```
data/
├── Formal_Settlement_Images/
│   ├── tile_001.tif
│   ├── tile_002.tif
│   └── ...
│
├── Informal_Settlemnt_Images/
│   ├── tile_001.tif
│   ├── tile_002.tif
│   └── ...
```

Each tile represents a 100m × 100m satellite image patch labeled as either:

- `Formal` (planned housing)
- `Informal` (slum / unplanned settlement)

Tiles are stored as GeoTIFF files.

# Indonesia Informal Settlement Classification

A deep learning pipeline for classifying satellite image tiles as either:

* **Formal Settlements** (planned housing)
* **Informal Settlements** (slum / unplanned housing)

using convolutional neural networks (CNNs) and geospatial satellite imagery.

---

# Project Structure

```text
Indonesia-Informal-Settlement-Classification/
│
├── config.yml
├── main.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── Formal_Settlement_Images/
│   └── Informal_Settlement_Images/
│
├── models/
│   ├── __init__.py
│   └── cnn.py
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   └── train.py
│
├── notebooks/
│
└── outputs/
    └── models/
```

---

# Dataset

The dataset consists of georeferenced satellite image tiles stored as `.tif` files.

Each tile represents a:

```text
100m × 100m satellite image patch
```

labeled as either:

| Label | Description         |
| ----- | ------------------- |
| 0     | Formal Settlement   |
| 1     | Informal Settlement |

---

# Requirements

* Python 3.10+
* pip
* virtual environment support (recommended)

---

# Setup Instructions

## 1. Clone Repository

```bash
git clone git@github.com:UCSD-Economics-Lab/Indonesia-Informal-Settlement-Classification.git

cd Indonesia-Informal-Settlement-Classification
```

---

## 2. Create Virtual Environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows (not tested)

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Pipeline
In the root of the directory: 
Run training with:

```bash
python3 main.py
```

The pipeline will:

1. Load satellite image tiles
2. Create train/test splits
3. Train a CNN classifier
4. Evaluate model performance
5. Save trained model weights

---
# Experiementing with the Pipeline
To change the models hyperparameter at a surface level you can change values in the config.yml file.
To change the model (not relevent now because we only have one) you can look for the model name and change the argument below

Run training with:

```bash
python3 main.py --config "config.yml" --model "cnn"
```
If you change the config.yml file name change the name in the arguments. Otherwise you dont need the argument since "config.yml" is the default. 


# Model Outputs

Trained models are saved to:

```text
outputs/models/
```

Example:

```text
outputs/models/cnn.pth
```

---

# Current Model

The current baseline model is a simple CNN implemented in:

```text
models/cnn.py
```


---

# Notes

* Satellite imagery is loaded using `rasterio`
* GeoTIFF (`.tif`) files may contain multiple spectral bands
* Current baseline uses the first 3 channels (RGB)



