"""
Custom samplers for image retrieval training.
"""

from collections import Counter
from torch.utils.data import WeightedRandomSampler, DataLoader
from pytorch_metric_learning.samplers import MPerClassSampler

def get_sampler(train_labels, batch_size, length_before_new_iter=None):
    """
    Get appropriate sampler for training data.
    
    This function analyzes the class distribution and selects the best sampler:
    - If there are enough classes and samples per class, uses MPerClassSampler (P-K strategy)
    - Otherwise, falls back to WeightedRandomSampler for balanced sampling
    """
    counts = Counter(train_labels)
    num_classes = len(counts)
    min_count = min(counts.values())
    
    # P-K sampling strategy parameters
    m_desired = 4                     # K = samples per class (4)
    P_desired = batch_size // m_desired  # P = number of classes per batch
    
    # If we have enough classes and samples per class, use MPerClassSampler
    if num_classes >= P_desired and min_count >= m_desired:
        print(f"Using MPerClassSampler: P={P_desired}, K={m_desired}")
        
        if length_before_new_iter is None:
            length_before_new_iter = len(train_labels)
            
        return MPerClassSampler(
            train_labels,
            m=m_desired,
            length_before_new_iter=length_before_new_iter,
            batch_size=batch_size
        )
    else:
        print(f"Dataset contains only {num_classes} classes or small classes (min {min_count} examples).")
        print("   → Using WeightedRandomSampler for balanced oversampling")
        
        # Calculate weights inversely proportional to class frequency
        weights = [1.0/counts[lbl] for lbl in train_labels]
        
        return WeightedRandomSampler(
            weights,
            num_samples=len(train_labels),
            replacement=True
        )

def get_dataloader(dataset, batch_size, sampler=None, shuffle=False, num_workers=4, drop_last=False):
    """
    Create a DataLoader with appropriate settings.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,  # sampler option is mutually exclusive with shuffle
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last
    )
