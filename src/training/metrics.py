"""
Evaluation metrics for image retrieval.
"""

import torch

def map_at_k(embeddings, labels, k=10):
    """
    Calculate mean Average Precision at k (mAP@k).
    
    Args:
        embeddings: L2-normalized embedding vectors (N, D)
        labels: Class labels for each embedding (N,)
        k: Number of top results to consider
        
    Returns:
        float: mAP@k score
    """
    # Calculate cosine similarity matrix
    sims = embeddings @ embeddings.t()
    
    # Set diagonal to -1 to exclude self-matches
    sims.fill_diagonal_(-1)
    
    # Get top-k indices for each query
    idx = sims.topk(k, dim=1).indices
    
    # Check which retrieved items match the query's class
    hits = (labels[idx] == labels.unsqueeze(1)).float()
    
    # Calculate precision at each position
    ranks = torch.arange(1, k+1, device=embeddings.device).float()
    prec = (hits.cumsum(dim=1) / ranks) * hits
    
    # Calculate average precision for each query
    ap = prec.sum(dim=1) / hits.sum(dim=1).clamp(min=1)
    
    # Return mean average precision
    return ap.mean().item()

def recall_at_k(embeddings, labels, k=10):
    """
    Calculate Recall at k (R@k).
    
    Args:
        embeddings: L2-normalized embedding vectors (N, D)
        labels: Class labels for each embedding (N,)
        k: Number of top results to consider
        
    Returns:
        float: R@k score
    """
    # Calculate cosine similarity matrix
    sims = embeddings @ embeddings.t()
    
    # Set diagonal to -1 to exclude self-matches
    sims.fill_diagonal_(-1)
    
    # Get top-k indices for each query
    topk = sims.topk(k, dim=1).indices
    
    # Check if any retrieved item matches the query's class
    hits = (labels[topk] == labels.unsqueeze(1)).any(1).float()
    
    # Return mean recall
    return hits.mean().item()

def precision_at_k(embeddings, labels, k=10):
    """
    Calculate Precision at k (P@k).
    
    Args:
        embeddings: L2-normalized embedding vectors (N, D)
        labels: Class labels for each embedding (N,)
        k: Number of top results to consider
        
    Returns:
        float: P@k score
    """
    # Calculate cosine similarity matrix
    sims = embeddings @ embeddings.t()
    
    # Set diagonal to -1 to exclude self-matches
    sims.fill_diagonal_(-1)
    
    # Get top-k indices for each query
    idx = sims.topk(k, dim=1).indices
    
    # Check which retrieved items match the query's class
    hits = (labels[idx] == labels.unsqueeze(1)).float()
    
    # Calculate precision for each query
    precision = hits.sum(dim=1) / k
    
    # Return mean precision
    return precision.mean().item()
