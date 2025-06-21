"""
Training pipeline for image retrieval.
"""

import math
import torch
import time
from pathlib import Path
from tqdm.auto import tqdm

from config import CFG, DEVICE, gpu_clear
from src.data.transforms import get_train_transforms, get_val_transforms
from src.data.loader import load_datasets, create_dataloaders
from src.training.model import RetrievalCLIP
from src.training.optimizer import create_optimizer_and_scheduler
from src.training.loss import get_loss_and_miner
from src.training.metrics import map_at_k, recall_at_k, precision_at_k

def train(args):
    """
    Run the complete training pipeline.
    
    Args:
        args: Arguments with batch_size, epochs, img_size, etc.
        
    Returns:
        Path: Path to best model checkpoint
    """
    print("\n=== TRAINING PIPELINE ===")
    
    # Create model config
    model_cfg = CFG.copy()
    model_cfg["batch_phys"] = args.batch_size
    model_cfg["num_epochs"] = args.epochs
    model_cfg["img_size"] = args.img_size
    
    # Create transforms
    train_transform = get_train_transforms(args.img_size)
    val_transform = get_val_transforms(args.img_size)
    
    # Load datasets
    datasets = load_datasets(train_transform, val_transform, val_split=0.1, seed=args.seed)
    print(f"Train: {len(datasets['train_ds'])} | Val: {len(datasets['val_ds'])}")
    
    # Create dataloaders
    loaders = create_dataloaders(datasets, args.batch_size, num_workers=args.num_workers)
    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]
    
    # Create model
    model = RetrievalCLIP(model_cfg).to(DEVICE)
    
    # Create optimizer and scheduler
    steps_per_epoch = math.ceil(len(datasets["train_ds"]) / (args.batch_size * CFG["grad_accum"]))
    optimizer, scheduler = create_optimizer_and_scheduler(model, model_cfg, steps_per_epoch)
    
    # Train model
    best_model_path = train_model(
        model, 
        optimizer, 
        scheduler, 
        train_loader, 
        val_loader, 
        model_cfg, 
        num_epochs=args.epochs
    )
    
    # Clean up GPU memory
    gpu_clear()
    
    return best_model_path

def train_model(model, optimizer, scheduler, train_loader, val_loader, cfg, num_epochs):
    """Train the model for the specified number of epochs."""
    # Initialize variables
    best_map = 0
    best_recall = 0
    no_improve = 0
    grad_accum = cfg["grad_accum"]
    scaler = torch.amp.GradScaler()
    best_model_path = Path("models/best_model.pth")
    
    # Calculate unfreeze epoch
    unfreeze_epoch = int(0.3 * num_epochs)
    
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        
        # Unfreeze backbone if it's time
        if epoch == unfreeze_epoch:
            model.unfreeze_all()
            print(f"--- Epoch {epoch}: backbone unfrozen")
            
        # Get loss function and miner for this epoch
        criterion, miner = get_loss_and_miner(epoch, cfg, DEVICE)
        
        # Training loop
        pbar = tqdm(enumerate(train_loader, 1), total=len(train_loader),
                    desc=f"Epoch {epoch}/{num_epochs}", leave=False)
        
        # Zero gradients at start
        optimizer.zero_grad(set_to_none=True)
        
        for step, (imgs, targets) in pbar:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                embeds = model(imgs)
                if miner is None:
                    loss = criterion(embeds, targets)
                else:
                    hard_pairs = miner(embeds, targets)
                    loss = criterion(embeds, targets, hard_pairs)
                loss = loss / grad_accum
                
            # Backward pass
            scaler.scale(loss).backward()
            running_loss += loss.item() * grad_accum
            
            # Update weights after accumulating gradients
            if step % grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        
        # Evaluate
        model.eval()
        val_metrics = evaluate(model, val_loader, cfg["k_top"])
        m, r, p = val_metrics["map"], val_metrics["recall"], val_metrics["precision"]
        
        print(f"Epoch {epoch:02d} | loss {running_loss/len(train_loader):.4f} | "
              f"mAP@{cfg['k_top']} {m:.4f} | R@{cfg['k_top']} {r:.4f} | "
              f"P@{cfg['k_top']} {p:.4f} | t{time.time()-t0:.1f}s")
        
        # Save backup model
        backup_path = Path(f"models/model_{epoch}.pth")
        torch.save(model.state_dict(), backup_path)
        print(f"     Backup model saved: {backup_path.name}")
        
        # Check if model improved
        improved = (m > best_map + 1e-4) or (abs(m - best_map) <= 1e-3 and r > best_recall)
        if improved:
            best_map, best_recall = m, r
            no_improve = 0
            
            # Save best model
            torch.save(model.state_dict(), best_model_path)
            print(f"     Best model saved")
        else:
            no_improve += 1
            if no_improve >= cfg["patience"]:
                print(f"Early stopping after {cfg['patience']} epochs without improvement")
                break
    
    print(f"Best mAP@{cfg['k_top']}={best_map:.4f}, Recall@{cfg['k_top']}={best_recall:.4f}")
    return best_model_path

def evaluate(model, val_loader, k=10):
    """Evaluate model on validation set."""
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        for imgs, targets in val_loader:
            embeddings = model(imgs.to(DEVICE))
            all_embeddings.append(embeddings.cpu())
            all_labels.append(targets)
    
    # Concatenate embeddings and labels
    embeddings = torch.cat(all_embeddings)
    labels = torch.cat(all_labels)
    
    # Calculate metrics
    m = map_at_k(embeddings, labels, k)
    r = recall_at_k(embeddings, labels, k)
    p = precision_at_k(embeddings, labels, k)
    
    return {"map": m, "recall": r, "precision": p}
