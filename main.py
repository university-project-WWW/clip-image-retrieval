#!/usr/bin/env python
"""
Image Retrieval Pipeline.
"""

import argparse
import torch
import sys
from pathlib import Path

# Aggiungi la directory corrente al path per risolvere problemi di importazione
sys.path.append('.')

# Importa dalla configurazione
from config import DEVICE

# Adatta le importazioni ai nomi di file esistenti
# Se hai rinominato i file, usa: from src.training.pipeline import train
# Altrimenti usa il nome effettivo del file:
from src.training.pipeline import train
from src.inference.pipeline import inference  # Modifica anche questo se necessario

def parse_args():
    parser = argparse.ArgumentParser(description="Image Retrieval Pipeline")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of epochs")
    parser.add_argument("--img-size", type=int, default=224,
                        help="Image size")
    parser.add_argument("--advanced", action="store_true",
                        help="Use advanced techniques (TTA, re-ranking, query expansion)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed")
    return parser.parse_args()

def setup():
    """Setup environment and directories."""
    args = parse_args()
    
    # Print device info
    if torch.cuda.is_available():
        print(f"✅ Device → {DEVICE} | GPU memory: "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    else:
        print(f"✅ Device → {DEVICE}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create directories
    Path("models").mkdir(exist_ok=True)
    Path("json_results").mkdir(exist_ok=True)
    
    return args

if __name__ == "__main__":
    args = setup()
    
    # Run training
    model_path = train(args)
    
    # Run inference
    if model_path and model_path.exists():
        inference(args, model_path)
    
    print("\n✅ Pipeline completed successfully!")