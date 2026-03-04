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

Each tile represents a 50m × 50m satellite image patch labeled as either:

- `Formal` (planned housing)
- `Informal` (slum / unplanned settlement)

Tiles are stored as GeoTIFF files.

