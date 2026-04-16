# train_central.py
"""
Centralized training — train model on all data combined.
Use this FIRST to verify the model works before FL.
Achieves ~85-90% accuracy as baseline.
"""
import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, 
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

from config import cfg
from model import ChestXRayModel, FocalLoss, get_model
from dataset import get_dataloaders
from utils import (
    save_checkpoint, load_checkpoint, 
    set_seed, EarlyStopping, AverageMeter
)

def train_one_epoch(
    model, loader, optimizer, criterion, 
    device, scaler, writer, epoch
):
    """
    Train for one epoch. Uses mixed precision (AMP) for speed.
    Returns average loss and accuracy.
    """
    model.train()
    losses = AverageMeter()
    correct = total = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", 
                leave=False, colour='green')
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Mixed precision forward pass
        with torch.cuda.amp.autocast(enabled=device=='cuda'):
            logits = model(images)
            loss = criterion(logits, labels)
        
        # Backward pass with gradient scaling
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # Gradient clipping (good practice even without DP)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )
        
        scaler.step(optimizer)
        scaler.update()
        
        # Metrics
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)
        losses.update(loss.item(), len(labels))
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{100*correct/total:.1f}%'
        })
        
        # Log every 50 batches
        step = epoch * len(loader) + batch_idx
        if step % 50 == 0:
            writer.add_scalar('Train/BatchLoss', loss.item(), step)
    
    epoch_acc = accuracy_score(all_labels, all_preds)
    writer.add_scalar('Train/Loss', losses.avg, epoch)
    writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
    
    return losses.avg, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device, writer, epoch):
    """Validation loop — no gradients, compute full metrics."""
    model.eval()
    losses = AverageMeter()
    all_preds = []
    all_labels = []
    all_probs = []
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val]", 
                leave=False, colour='blue')
    
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        probs = torch.softmax(logits, dim=1)[:, 1]  # P(Pneumonia)
        preds = logits.argmax(dim=1)
        
        losses.update(loss.item(), len(labels))
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    
    writer.add_scalar('Val/Loss', losses.avg, epoch)
    writer.add_scalar('Val/Accuracy', acc, epoch)
    writer.add_scalar('Val/AUC-ROC', auc, epoch)
    
    print(f"\n  Val Loss: {losses.avg:.4f} | "
          f"Accuracy: {100*acc:.2f}% | AUC-ROC: {auc:.4f}")
    print(classification_report(
        all_labels, all_preds, 
        target_names=cfg.class_names, digits=4
    ))
    
    return losses.avg, acc, auc


def train():
    set_seed(42)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    
    print(f"[Config] Device: {cfg.device}")
    print(f"[Config] Classes: {cfg.class_names}")
    print(f"[Config] Epochs: {cfg.num_epochs}")
    
    # Data
    train_loader, val_loader = get_dataloaders()
    
    # Model
    model = get_model(cfg.device)
    total, trainable = model.get_parameter_count()
    print(f"[Model] Total: {total:,} | Trainable: {trainable:,}")
    
    # Loss — Focal loss handles class imbalance better
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Optimizer — AdamW with weight decay
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # LR Scheduler — OneCycleLR for fast convergence
    scheduler = OneCycleLR(
        optimizer, max_lr=cfg.learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=cfg.num_epochs,
        pct_start=0.3   # 30% warmup
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.device=='cuda')
    
    # TensorBoard
    writer = SummaryWriter(cfg.log_dir)
    
    # Early stopping
    early_stop = EarlyStopping(
        patience=cfg.early_stopping_patience,
        min_delta=0.001,
        mode='max'    # Monitor validation accuracy
    )
    
    best_auc = 0.0
    
    for epoch in range(cfg.num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{cfg.num_epochs}")
        
        # Unfreeze backbone after warmup (epoch 5)
        if epoch == 5:
            model.unfreeze_backbone(layers=3)
            # Rebuild optimizer with new params + lower LR for backbone
            optimizer = torch.optim.AdamW([
                {'params': model.features.parameters(), 
                 'lr': cfg.learning_rate * 0.1},  # Lower LR for backbone
                {'params': model.classifier.parameters(), 
                 'lr': cfg.learning_rate}
            ], weight_decay=cfg.weight_decay)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion,
            cfg.device, scaler, writer, epoch
        )
        
        # Validate
        val_loss, val_acc, val_auc = validate(
            model, val_loader, criterion, 
            cfg.device, writer, epoch
        )
        
        scheduler.step()
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_acc': val_acc,
                'config': cfg,
            }, path=cfg.global_model_path)
            print(f"  ✓ New best AUC: {val_auc:.4f} — checkpoint saved")
        
        # Early stopping check
        if early_stop(val_acc):
            print(f"\n[EarlyStopping] No improvement for "
                  f"{cfg.early_stopping_patience} epochs. Stopping.")
            break
    
    writer.close()
    print(f"\n[Training Complete] Best Val AUC: {best_auc:.4f}")
    print(f"Model saved to: {cfg.global_model_path}")


if __name__ == "__main__":
    train()
