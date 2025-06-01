"""
Loss functions and miners for image retrieval training.
"""

from pytorch_metric_learning.losses import CircleLoss
from pytorch_metric_learning.miners import MultiSimilarityMiner, TripletMarginMiner

def get_loss_and_miner(epoch, cfg, device):
    """
    Get appropriate loss function and miner based on training epoch.
    
    Implementation follows a curriculum strategy:
    - Early epochs (1-4): Only CircleLoss without mining
    - Later epochs (5+): CircleLoss with MultiSimilarity mining
    
    Args:
        epoch: Current training epoch
        cfg: Configuration dictionary with loss parameters
        device: Device to place the loss function on
        
    Returns:
        tuple: (loss_function, miner)
    """
    # Create base Circle Loss with configured parameters
    base_circle = CircleLoss(
        m=cfg["loss_margin"],     # Margin parameter
        gamma=cfg["loss_gamma"]   # Scale parameter
    )
    
    # For early epochs, use no mining to establish initial feature space
    if epoch < 5:
        return base_circle.to(device), None
    
    # For later epochs, use MultiSimilarityMiner to find hard examples
    else:
        # MultiSimilarity miner finds both hard positives and hard negatives
        multisim_miner = MultiSimilarityMiner(epsilon=0.1)
        return base_circle.to(device), multisim_miner
