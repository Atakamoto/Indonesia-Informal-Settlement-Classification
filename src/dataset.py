from pathlib import Path

from jupyter_core.migrate import src
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize


class SettlementDataset(Dataset):
    def __init__(self, image_paths, labels, image_size=224, band_indices=[1, 2, 3]):
        self.image_paths = image_paths
        self.labels = labels
        self.image_size = image_size
        self.band_indices = band_indices
        self.resize = Resize((image_size, image_size))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        with rasterio.open(path) as src:
            image = src.read(self.band_indices)

        image = image[:self.input_channels, :, :]

        image = image.astype(np.float32)

        # Normalize uint8 imagery from 0-255 to 0-1
        if image.max() > 1:
            image = image / 255.0

        image = torch.tensor(image, dtype=torch.float32)
        image = self.resize(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label


def load_image_paths(formal_dir, informal_dir):
    formal_paths = list(Path(formal_dir).glob("*.tif"))
    informal_paths = list(Path(informal_dir).glob("*.tif"))

    image_paths = formal_paths + informal_paths

    # 0 = formal, 1 = informal
    labels = [0] * len(formal_paths) + [1] * len(informal_paths)

    return image_paths, labels