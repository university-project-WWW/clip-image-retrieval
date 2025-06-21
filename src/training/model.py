"""
Network architecture for image retrieval.
Uses Vision Transformer with CLIP pretraining as backbone.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from einops import rearrange

# Custom GeM (Generalized Mean) Pooling implementation
class GeM(nn.Module):
    """
    Generalized Mean Pooling as described in the paper:
    "Fine-tuning CNN Image Retrieval with No Human Annotation"
    """
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # Apply power p
        x = x.clamp(min=self.eps).pow(self.p)
        # Pool spatially
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        # Apply inverse power 1/p
        return x.pow(1. / self.p)

class RetrievalCLIP(nn.Module):
    """
    Image retrieval model based on CLIP ViT with GeM pooling.
    
    Architecture:
    1. CLIP ViT-L/14 backbone (frozen by default)
    2. GeM pooling to aggregate token features
    3. Linear projection to embedding dimension
    4. Batch normalization and dropout
    5. L2 normalization of final embeddings
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Backbone: OpenCLIP ViT-L/14
        self.backbone = timm.create_model(
            "vit_large_patch14_clip_224",  # CLIP with 14x14 patches
            pretrained=True,
            num_classes=0,
            global_pool=""
        )
        
        # Enable gradient checkpointing to save memory
        self.backbone.set_grad_checkpointing()
        
        # Feature dimensions (1024 for CLIP ViT-L/14)
        hid_dim = self.backbone.num_features  # 1024
        
        # With 14x14 patches, the feature grid is 16x16
        self.hid_h = self.hid_w = int(math.sqrt(self.backbone.patch_embed.num_patches))  # 16
        
        # Custom GeM pooling implementation
        self.pool = GeM(p=3)
        
        # Embedding projection head
        self.fc = nn.Linear(hid_dim, cfg["emb_dim"], bias=False)
        self.bn = nn.BatchNorm1d(cfg["emb_dim"])
        self.do = nn.Dropout(0.1)  # Reduced from 0.2 to 0.1 (CLIP has internal regularization)

    def forward(self, x):
        """Forward pass of the model."""
        # Extract token features (B, 1+N, C) - N patch tokens plus CLS token
        tokens = self.backbone.forward_features(x)
        
        # Discard CLS token and reshape to feature map (B, C, H, W)
        fmap = tokens[:, 1:, :]
        fmap = rearrange(fmap, "b (h w) c -> b c h w", h=self.hid_h, w=self.hid_w)
        
        # GeM pooling + flatten
        pooled = self.pool(fmap).flatten(1)
        
        # Project to embedding dimension
        z = self.fc(pooled)
        z = self.bn(z)
        z = self.do(z)
        
        # L2 normalization for cosine similarity
        return F.normalize(z, dim=1)  # (B, emb_dim)

