"""
Similarity search for image retrieval.
"""

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from src.inference.rerank import re_ranking

def cosine_similarity_search(query_embeddings, gallery_embeddings, k=10, chunk_size=2048):
    """
    Perform cosine similarity search in chunks to reduce memory usage.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gallery_embeddings_t = gallery_embeddings.T.to(device)  # (D, G)
    
    topk_indices = []
    
    # Process queries in chunks to avoid OOM
    for start in tqdm(range(0, len(query_embeddings), chunk_size), desc="Similarity Search"):
        end = min(start + chunk_size, len(query_embeddings))
        q_chunk = query_embeddings[start:end].to(device)  # (chunk, D)
        
        # Calculate similarity matrix for this chunk
        # sim (chunk, G) = q_chunk @ gallery.T
        sim_chunk = torch.mm(q_chunk.float(), gallery_embeddings_t.float())
        
        # Get top-k indices
        _, indices = sim_chunk.topk(k=k, dim=1, largest=True)
        
        topk_indices.append(indices.cpu())
    
    # Concatenate results from all chunks
    topk_indices = torch.cat(topk_indices, dim=0)  # (Q, k)
    
    return topk_indices

def query_expansion(query_embeddings, gallery_embeddings, initial_indices, alpha=0.3, k_expand=3):
    """
    Apply query expansion to improve retrieval results.
    
    Query expansion averages the query with its top-k neighbors to create an expanded query.
    """
    # Get top-k gallery embeddings for each query
    top_k_gallery = gallery_embeddings[initial_indices[:, :k_expand]]  # (Q, k_expand, D)
    
    # Average the top-k gallery embeddings
    top_k_avg = top_k_gallery.mean(dim=1)  # (Q, D)
    
    # Combine original query with top-k average
    expanded_queries = alpha * query_embeddings + (1 - alpha) * top_k_avg
    
    # Re-normalize expanded queries
    expanded_queries = F.normalize(expanded_queries, dim=1)
    
    return expanded_queries

def reranking(query_embeddings, gallery_embeddings, k1=20, k2=6, lambda_value=0.3):
    """
    Apply re-ranking to improve retrieval results.
    
    Re-ranking refines initial results by considering both query-to-gallery
    and gallery-to-gallery relationships.
    """
    # Compute distance matrices (1 - cosine similarity)
    qg_dist = (1.0 - (query_embeddings @ gallery_embeddings.T)).numpy().astype('float32')
    qq_dist = (1.0 - (query_embeddings @ query_embeddings.T)).numpy().astype('float32')
    gg_dist = (1.0 - (gallery_embeddings @ gallery_embeddings.T)).numpy().astype('float32')
    
    # Apply re-ranking
    rerank_dist = re_ranking(
        q_g_dist=qg_dist,
        q_q_dist=qq_dist,
        g_g_dist=gg_dist,
        k1=k1,
        k2=k2,
        lambda_value=lambda_value
    )
    
    # Convert distance to similarity (1 - distance)
    rerank_sim = 1.0 - rerank_dist
    
    # Get top-k indices based on re-ranked similarities
    topk_values, topk_indices = torch.from_numpy(rerank_sim).topk(
        k=min(k1, gallery_embeddings.size(0)), 
        dim=1, 
        largest=True
    )
    
    return topk_indices

def advanced_retrieval(query_embeddings, gallery_embeddings, cfg):
    """
    Perform advanced retrieval with re-ranking and query expansion.
    
    Process:
    1. Initial retrieval with cosine similarity
    2. Re-ranking to refine results
    3. Query expansion with top results
    4. Final retrieval with expanded queries
    """
    print("Performing advanced retrieval with re-ranking and query expansion...")
    
    # Step 1: Initial retrieval
    initial_indices = cosine_similarity_search(
        query_embeddings, 
        gallery_embeddings, 
        k=cfg["rerank_k1"]
    )
    
    # Step 2: Re-ranking
    print("Applying re-ranking...")
    reranked_indices = reranking(
        query_embeddings,
        gallery_embeddings,
        k1=cfg["rerank_k1"],
        k2=cfg["rerank_k2"],
        lambda_value=cfg["rerank_lambda"]
    )
    
    # Step 3: Query expansion using re-ranked results
    print("Applying query expansion...")
    expanded_queries = query_expansion(
        query_embeddings,
        gallery_embeddings,
        reranked_indices,
        alpha=cfg["qe_alpha"],
        k_expand=3
    )
    
    # Step 4: Final retrieval with expanded queries
    print("Final retrieval with expanded queries...")
    final_indices = cosine_similarity_search(
        expanded_queries,
        gallery_embeddings,
        k=cfg["k_top"]
    )
    
    return final_indices
