"""
Re-ranking implementation for image retrieval based on k-reciprocal encoding.

Reference:
Zhong Z, Zheng L, Cao D, Li S. 
"Re-ranking Person Re-identification with k-reciprocal Encoding"
In CVPR 2017
"""

import numpy as np

def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    """
    Re-ranking based on k-reciprocal encoding.
    
    Args:
        q_g_dist: Query-gallery distance matrix (query count x gallery count)
        q_q_dist: Query-query distance matrix (query count x query count)
        g_g_dist: Gallery-gallery distance matrix (gallery count x gallery count)
        k1: Number of neighbors for the k-reciprocal set
        k2: Number of neighbors for the extended jaccard set
        lambda_value: Weighting parameter (0-1)
        
    Returns:
        numpy.ndarray: Re-ranked distance matrix
    """
    # Original feature distance
    original_dist = np.concatenate(
        [
            np.concatenate([q_q_dist, q_g_dist], axis=1),
            np.concatenate([q_g_dist.T, g_g_dist], axis=1)
        ], 
        axis=0
    )
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = 1.0 - original_dist
    
    # Shape of the distance matrix
    all_num = original_dist.shape[0]
    q_num = q_q_dist.shape[0]
    g_num = g_g_dist.shape[0]
    
    # Create the final distance matrix
    final_dist = np.zeros_like(original_dist)
    
    # For each element in the joint query-gallery set
    for i in range(all_num):
        # Get k1+1 nearest neighbors (including self)
        initial_rank = np.argsort(original_dist[i])
        forward_k_neigh_index = initial_rank[:k1+1]
        
        # Create k-reciprocal set
        k_reciprocal_index = forward_k_neigh_index.copy()
        for j in range(len(forward_k_neigh_index)):
            candidate = forward_k_neigh_index[j]
            # Get k1+1 nearest neighbors of the candidate
            candidate_forward_k_neigh_index = np.argsort(original_dist[candidate])[:k1+1]
            # Check if i is in the k1-nearest neighbors of j
            if np.where(candidate_forward_k_neigh_index == i)[0].size > 0:
                k_reciprocal_index = np.append(k_reciprocal_index, candidate_forward_k_neigh_index)
        
        # Remove duplicates and self
        k_reciprocal_index = np.unique(k_reciprocal_index)
        weight = np.exp(-original_dist[i, k_reciprocal_index])
        
        # Compute Jaccard distance
        dist = np.zeros(all_num)
        for j in range(len(k_reciprocal_index)):
            idx = k_reciprocal_index[j]
            # Get k2+1 nearest neighbors for extended Jaccard
            neighbor_idx = np.argsort(original_dist[idx])[:k2+1]
            
            # Compute weighted sum
            for n_idx in neighbor_idx:
                dist[n_idx] += weight[j] * original_dist[idx, n_idx] / (k2 + 1)
        
        # Set zero for k-reciprocal neighbors
        dist[k_reciprocal_index] = 0
        final_dist[i] = dist
    
    # Get the minimum distance for symmetry
    for i in range(all_num):
        for j in range(all_num):
            final_dist[i, j] = max(final_dist[i, j], final_dist[j, i])
    
    # Apply lambda weighting between original and re-ranked distances
    final_dist = lambda_value * final_dist + (1 - lambda_value) * original_dist
    
    # Extract the query-gallery part for the result
    re_ranked_dist = final_dist[:q_num, q_num:]
    
    return re_ranked_dist
