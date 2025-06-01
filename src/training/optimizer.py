"""
Optimizer and learning rate scheduler for image retrieval training.
Implements Layer-wise Learning Rate Decay (LLRD) for ViT models.
"""

import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

def get_layerwise_params(model, cfg):
    """
    Implement Layer-wise Learning Rate Decay (LLRD) for ViT models.
    
    The approach assigns different learning rates to different layers:
    - Head parameters (non-backbone): Highest learning rate
    - Transformer blocks: Decaying learning rate from top to bottom
    - Other backbone parameters (embeddings, etc.): Very low learning rate
    """
    base_lr = cfg["lr_backbone"]  # 4e-4 (from paper)
    head_lr = cfg["lr_head"]      # 4e-4 (same as backbone in paper)
    decay_factor = 0.65           # From paper for ViT-L
    
    param_groups = []
    assigned_params = set()  # Track already assigned parameters
    
    # 1. Head parameters (non-backbone) - fc, bn, do
    head_params = []
    for name, param in model.named_parameters():
        if "backbone" not in name:
            head_params.append(param)
            assigned_params.add(id(param))
    
    if head_params:
        param_groups.append({
            "params": head_params, 
            "lr": head_lr, 
            "weight_decay": cfg["weight_decay"],
            "name": "head"
        })
    
    # 2. Backbone transformer blocks with layer-wise decay
    num_layers = len(model.backbone.blocks)  # ~24 for ViT-L/14
    
    for layer_idx in range(num_layers):
        layer_lr = base_lr * (decay_factor ** (num_layers - layer_idx - 1))
        layer_params = []
        
        for name, param in model.named_parameters():
            if f"backbone.blocks.{layer_idx}." in name:  # Add dot to avoid false matches
                if id(param) not in assigned_params:
                    layer_params.append(param)
                    assigned_params.add(id(param))
        
        if layer_params:
            param_groups.append({
                "params": layer_params, 
                "lr": layer_lr, 
                "weight_decay": cfg["weight_decay"],
                "name": f"layer_{layer_idx}"
            })
    
    # 3. Other backbone parameters (embeddings, norms, etc.)
    other_backbone_params = []
    for name, param in model.named_parameters():
        if "backbone" in name and "blocks" not in name:
            if id(param) not in assigned_params:
                other_backbone_params.append(param)
                assigned_params.add(id(param))
    
    if other_backbone_params:
        param_groups.append({
            "params": other_backbone_params, 
            "lr": base_lr * 0.1,  # Very low LR for embeddings
            "weight_decay": cfg["weight_decay"],
            "name": "backbone_other"
        })
    
    # Verify all parameters are assigned
    total_model_params = set(id(p) for p in model.parameters())
    if len(assigned_params) != len(total_model_params):
        print(f"Warning: {len(total_model_params) - len(assigned_params)} parameters not assigned!")
        
        # Print unassigned parameters for debugging
        for name, param in model.named_parameters():
            if id(param) not in assigned_params:
                print(f"   Not assigned: {name}")
    
    return param_groups

def create_optimizer_and_scheduler(model, cfg, steps_per_epoch):
    """
    Create optimizer with LLRD and cosine learning rate scheduler with warmup.
    """
    # Get parameter groups with layer-wise learning rates
    optimizer_params = get_layerwise_params(model, cfg)
    
    # Create AdamW optimizer
    optimizer = torch.optim.AdamW(optimizer_params)
    
    # Calculate total steps and warmup steps
    total_steps = steps_per_epoch * cfg["num_epochs"]
    warmup_steps = int(0.15 * total_steps)  # 15% warmup
    
    # 1) Linear warmup: LR_init = LR_max / 25 → LR_max
    scheduler_warm = LinearLR(
        optimizer,
        start_factor=1 / 25,     # 0.04 (>= 0.0 and <= 1.0)
        end_factor=1.0,          # Reach base LR
        total_iters=warmup_steps
    )
    
    # 2) Cosine annealing: LR_max → eta_min
    scheduler_cos = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=4e-9  # Very small LR at the end
    )
    
    # 3) Chain warmup + cosine schedulers
    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_warm, scheduler_cos],
        milestones=[warmup_steps]
    )
    
    return optimizer, scheduler
