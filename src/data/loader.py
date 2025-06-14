"""
Data loading functions for image retrieval.
"""

import torch
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset

from config import TRAIN_DIR, QUERY_DIR, GALLERY_DIR
from src.data.datasets import FlatFolderDataset
from src.data.samplers import get_sampler, get_dataloader
from src.data.transforms import get_train_transforms, get_val_transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split


def load_datasets(train_transform, val_transform, val_split=0.1, seed=123):
    """
    Load all datasets for the retrieval task.
    
    Args:
        train_transform: Transforms to apply to training images
        val_transform: Transforms to apply to validation/test images
        val_split: Fraction of training data to use for validation
        seed: Random seed for reproducibility
        
    Returns:
        dict: Dictionary containing all datasets and related info
    """
    # Load full training set
    full_train = ImageFolder(TRAIN_DIR, transform=train_transform)
    labels = np.array([y for _, y in full_train.imgs])
    
    # Split into train and validation

    # StratifiedShuffle removed in competition (due to dataset structure)
    # sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    # train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))

    #Version for competition:
    train_idx, val_idx = train_test_split(
        np.arange(len(labels)),
        test_size=0.10,
        random_state=seed,
        stratify=None  # niente stratificazione
    )
    
    # Create train and validation datasets
    train_ds = Subset(full_train, train_idx)
    val_ds = Subset(ImageFolder(TRAIN_DIR, transform=val_transform), val_idx)
    
    # Create query and gallery datasets
    query_ds = FlatFolderDataset(QUERY_DIR, transform=val_transform)
    gallery_ds = FlatFolderDataset(GALLERY_DIR, transform=val_transform)
    
    # Calculate class distribution for sampler configuration
    train_labels = [full_train.targets[i] for i in train_idx]
    
    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "query_ds": query_ds,
        "gallery_ds": gallery_ds,
        "train_labels": train_labels,
        "num_classes": len(full_train.classes),
        "class_to_idx": full_train.class_to_idx
    }

def create_dataloaders(datasets, batch_size, num_workers=4):
    """
    Create dataloaders for all datasets.
    
    Args:
        datasets: Dictionary with datasets from load_datasets()
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        
    Returns:
        dict: Dictionary containing all dataloaders
    """
    # Create train sampler
    train_sampler = get_sampler(
        datasets["train_labels"],
        batch_size,
        length_before_new_iter=len(datasets["train_ds"])
    )
    
    # Create train dataloader
    train_loader = get_dataloader(
        datasets["train_ds"],
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        drop_last=True
    )
    
    # Create validation dataloader
    val_loader = get_dataloader(
        datasets["val_ds"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    # Create query dataloader
    query_loader = get_dataloader(
        datasets["query_ds"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    # Create gallery dataloader
    gallery_loader = get_dataloader(
        datasets["gallery_ds"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "query_loader": query_loader,
        "gallery_loader": gallery_loader
    }

def load_query_gallery(val_transform, batch_size, num_workers=4):
    """
    Load only query and gallery datasets and create dataloaders.
    Used for inference only.
    
    Args:
        val_transform: Transforms to apply to images
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        
    Returns:
        tuple: (query_loader, gallery_loader, query_ds, gallery_ds)
    """
    # Create query and gallery datasets
    query_ds = FlatFolderDataset(QUERY_DIR, transform=val_transform)
    gallery_ds = FlatFolderDataset(GALLERY_DIR, transform=val_transform)
    
    # Create dataloaders
    query_loader = get_dataloader(
        query_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    gallery_loader = get_dataloader(
        gallery_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return query_loader, gallery_loader, query_ds, gallery_ds
