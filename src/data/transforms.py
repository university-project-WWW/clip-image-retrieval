"""
Data transformations for image retrieval.
"""

from torchvision import transforms
from config import MEAN, STD

def get_train_transforms(img_size=224):
    """
    Returns transformations for training images.
    Includes various data augmentations for improved model robustness.
    """
    return transforms.Compose([
        # Resize and crop with scale jitter (0.8 to 1.0)
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        # Horizontal flip with 50% probability
        transforms.RandomHorizontalFlip(0.5),
        # Color jitter for brightness, contrast, and saturation
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.0),
        # Random grayscale with 10% probability
        transforms.RandomGrayscale(0.1),
        # Convert to tensor
        transforms.ToTensor(),
        # Random erasing with 30% probability
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), value="random"),
        # Normalize with ImageNet mean and std
        transforms.Normalize(MEAN, STD)
    ])

def get_val_transforms(img_size=224):
    """
    Returns transformations for validation/test images.
    Simple resize and center crop without augmentations.
    """
    return transforms.Compose([
        # Resize to slightly larger size for center crop
        transforms.Resize(int(img_size * 256 / 224)),
        # Center crop to final size
        transforms.CenterCrop(img_size),
        # Convert to tensor
        transforms.ToTensor(),
        # Normalize with ImageNet mean and std
        transforms.Normalize(MEAN, STD)
    ])
