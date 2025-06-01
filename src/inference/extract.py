"""
Feature extraction for image retrieval.
"""

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

def extract_embeddings(model, data_loader, device, use_amp=True):
    """
    Extract embeddings from images using the model.
    
    Args:
        model: Model to use for extraction
        data_loader: DataLoader with images
        device: Device to run extraction on
        use_amp: Whether to use mixed precision
        
    Returns:
        tuple: (embeddings, filenames)
    """
    model.eval()
    all_embeddings = []
    all_filenames = []
    
    with torch.no_grad():
        for imgs, filenames in tqdm(data_loader, desc="Extract", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            
            # Use mixed precision for faster inference
            if use_amp:
                with torch.cuda.amp.autocast():
                    embeddings = model(imgs)
            else:
                embeddings = model(imgs)
            
            all_embeddings.append(embeddings.cpu())
            all_filenames.extend(filenames)
    
    # Concatenate embeddings from all batches
    embeddings = torch.cat(all_embeddings, dim=0)
    
    return embeddings, all_filenames

def extract_embeddings_with_tta(model, data_loader, device):
    """
    Extract embeddings with Test-Time Augmentation (TTA).
    Uses both original and horizontally flipped images.
    
    Args:
        model: Model to use for extraction
        data_loader: DataLoader with images
        device: Device to run extraction on
        
    Returns:
        tuple: (embeddings, filenames)
    """
    model.eval()
    all_embeddings = []
    all_filenames = []
    
    with torch.no_grad():
        for imgs, filenames in tqdm(data_loader, desc="TTA Extract", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                # Original image
                emb_orig = model(imgs)
                
                # Horizontally flipped image
                emb_flip = model(torch.flip(imgs, dims=[3]))
                
                # Average and re-normalize
                emb = F.normalize(emb_orig + emb_flip, dim=1)
            
            all_embeddings.append(emb.cpu())
            all_filenames.extend(filenames)
    
    # Concatenate embeddings from all batches
    embeddings = torch.cat(all_embeddings, dim=0)
    
    return embeddings, all_filenames
