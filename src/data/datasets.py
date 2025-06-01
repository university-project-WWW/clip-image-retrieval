"""
Dataset classes for image retrieval.
"""

import torch
from pathlib import Path
from PIL import Image

class FlatFolderDataset(torch.utils.data.Dataset):
    """
    Dataset for loading images from a flat directory (no class subfolders).
    Used for query and gallery sets.
    
    Args:
        root: Path to directory containing images
        transform: Optional transforms to apply to images
    """
    def __init__(self, root: Path, transform=None):
        self.files = sorted([p for p in Path(root).iterdir()
                           if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        assert self.files, f"No images found in {root}"
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, path.name
