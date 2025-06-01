"""
Inference pipeline for image retrieval.
"""

import torch
from pathlib import Path

from config import CFG, DEVICE, gpu_clear, SUBM_FILE
from src.data.transforms import get_val_transforms
from src.data.loader import load_query_gallery
from src.training.model import RetrievalCLIP
from src.inference.extract import extract_embeddings, extract_embeddings_with_tta
from src.inference.retrieval import cosine_similarity_search, advanced_retrieval
from src.inference.submission import create_submission, create_submission_with_tta

def inference(args, model_path):
    """
    Run the complete inference pipeline.
    
    Args:
        args: Arguments with batch_size, img_size, advanced, etc.
        model_path: Path to model checkpoint
        
    Returns:
        Path: Path to submission file
    """
    print("\n=== INFERENCE PIPELINE ===")
    
    # Create model config
    model_cfg = CFG.copy()
    model_cfg["batch_phys"] = args.batch_size
    model_cfg["img_size"] = args.img_size
    
    # Create transforms
    val_transform = get_val_transforms(args.img_size)
    
    # Load query and gallery datasets
    query_loader, gallery_loader, query_ds, gallery_ds = load_query_gallery(
        val_transform, args.batch_size, num_workers=args.num_workers
    )
    
    print(f"Query: {len(query_ds)} | Gallery: {len(gallery_ds)}")
    
    # Create and load model
    model = RetrievalCLIP(model_cfg).to(DEVICE)
    
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return None
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"✅ Model loaded from {model_path}")
    
    # Extract embeddings
    print("Extracting embeddings...")
    
    # Use TTA if advanced mode is enabled
    if args.advanced:
        query_embeddings, query_filenames = extract_embeddings_with_tta(
            model, query_loader, DEVICE
        )
        print(f"Query embeddings with TTA: {query_embeddings.shape}")
    else:
        query_embeddings, query_filenames = extract_embeddings(
            model, query_loader, DEVICE
        )
        print(f"Query embeddings: {query_embeddings.shape}")
    
    gallery_embeddings, gallery_filenames = extract_embeddings(
        model, gallery_loader, DEVICE
    )
    print(f"Gallery embeddings: {gallery_embeddings.shape}")
    
    # Perform retrieval
    print("Performing retrieval...")
    
    if args.advanced:
        # Advanced retrieval with TTA, re-ranking, and query expansion
        topk_indices = advanced_retrieval(
            query_embeddings, gallery_embeddings, model_cfg
        )
        
        # Create submission
        create_submission_with_tta(
            query_filenames, gallery_filenames, topk_indices,
            SUBM_FILE, k=CFG["k_top"], technique="advanced"
        )
    else:
        # Standard retrieval with cosine similarity
        topk_indices = cosine_similarity_search(
            query_embeddings, gallery_embeddings, k=CFG["k_top"]
        )
        
        # Create submission
        create_submission(
            query_filenames, gallery_filenames, topk_indices,
            SUBM_FILE, k=CFG["k_top"]
        )
    
    # Clean up
    gpu_clear()
    
    return SUBM_FILE
