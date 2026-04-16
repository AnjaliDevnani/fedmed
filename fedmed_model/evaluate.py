import torch
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import numpy as np
from tqdm import tqdm

from config import cfg
from model import get_model, FocalLoss
from dataset import get_dataloaders
from utils import AverageMeter

@torch.no_grad()
def evaluate_global_model():
    print(f"Loading global model from {cfg.global_model_path}...")
    model = get_model()
    checkpoint = torch.load(cfg.global_model_path, map_location=cfg.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    _, val_loader = get_dataloaders()
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    losses = AverageMeter()
    all_preds = []
    all_labels = []
    all_probs = []
    
    pbar = tqdm(val_loader, desc="Evaluating", leave=False)
    for images, labels in pbar:
        images = images.to(cfg.device, non_blocking=True)
        labels = labels.to(cfg.device, non_blocking=True)
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)
        
        losses.update(loss.item(), len(labels))
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    
    print(f"\\nEvaluation Results:")
    print(f"Loss: {losses.avg:.4f}")
    print(f"Accuracy: {100*acc:.2f}%")
    print(f"AUC-ROC: {auc:.4f}")
    print("\\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=cfg.class_names, digits=4))

if __name__ == "__main__":
    evaluate_global_model()
