"""
Configuration settings for the image retrieval system.
Contains paths, hyperparameters, and training settings.
"""

import os
import random
import torch
import numpy as np
from pathlib import Path

# -----------------------------------------------------------------------------#
# PATHS (Modify as needed based on your dataset structure)
# -----------------------------------------------------------------------------#
ROOT_DIR = Path("food101_mini_split")  # Main dataset directory
TRAIN_DIR = ROOT_DIR / "training"           # Training directory with class subfolders
QUERY_DIR = ROOT_DIR / "test" / "query"     # Query directory (flat structure)
GALLERY_DIR = ROOT_DIR / "test" / "gallery" # Gallery directory (flat structure)
SUBM_FILE = Path.cwd() / "retrieval_results.json"  # Output file for submission

# Output directories
MODELS_DIR = Path("models")
JSON_DIR = Path("json_results")

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
JSON_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------#
# DEVICE CONFIGURATION
# -----------------------------------------------------------------------------#
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CUDA memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.6"

def gpu_clear():
    """Clear CUDA cache to free up memory."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# -----------------------------------------------------------------------------#
# HYPERPARAMETERS
# -----------------------------------------------------------------------------#
CFG = {
    # Data processing
    "img_size": 224,                # Input image size
    "batch_phys": 128,              # Physical batch size
    "grad_accum": 2,                # Gradient accumulation steps
    
    # Training settings
    "num_epochs": 15,               # Maximum number of epochs
    "lr_backbone": 4e-4,            # Learning rate for backbone (from paper)
    "lr_head": 4e-4,                # Learning rate for head (from paper)
    "weight_decay": 0.05,           # Weight decay (from paper)
    "patience": 5,                  # Early stopping patience
    
    # Loss function parameters
    "loss_gamma": 80,               # Circle loss gamma parameter
    "loss_margin": 0.4,             # Circle loss margin parameter
    
    # Model parameters
    "emb_dim": 512,                 # Embedding dimension
    
    # Retrieval parameters
    "k_top": 10,                    # Number of retrieved images
    
    # Re-ranking parameters
    "rerank_k1": 20,                # k1 parameter for re-ranking
    "rerank_k2": 6,                 # k2 parameter for re-ranking
    "rerank_lambda": 0.25,          # Lambda parameter for re-ranking
    
    # Query expansion parameters
    "qe_alpha": 0.75,               # Alpha parameter for query expansion
}

# Image normalization parameters (ImageNet mean and std)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Print config info
def print_config():
    """Print main configuration settings."""
    print("\n=== CONFIGURATION ===")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"Dataset: {ROOT_DIR}")
    print(f"Image Size: {CFG['img_size']}")
    print(f"Batch Size: {CFG['batch_phys']} (physical) × {CFG['grad_accum']} (accumulation)")
    print(f"Effective Batch Size: {CFG['batch_phys'] * CFG['grad_accum']}")
    print(f"Learning Rate: {CFG['lr_backbone']}")
    print(f"Embedding Dimension: {CFG['emb_dim']}")
    print(f"Top-k: {CFG['k_top']}")
