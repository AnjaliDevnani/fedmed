# fl_client.py
"""
Flower Federated Learning Client.
Runs at each hospital. Trains locally, shares only gradients.
Implements Differential Privacy via Opacus.

Usage:
  python fl_client.py --hospital A
  python fl_client.py --hospital B
  python fl_client.py --hospital C
"""
import argparse
import torch
import torch.nn as nn
import numpy as np
import flwr as fl
from flwr.common import NDArrays, Scalar
from typing import Dict, List, Tuple
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from config import cfg
from model import ChestXRayModel, FocalLoss, get_model
from dataset import get_dataloaders
from utils import set_seed, AverageMeter


class HospitalClient(fl.client.NumPyClient):
    """
    Federated Learning client representing one hospital.
    
    Each round:
    1. Receives global model weights from FL server
    2. Trains locally on hospital's own patient data
       (with Differential Privacy — Opacus adds noise)
    3. Returns updated weights + metrics to FL server
    
    CRITICAL: Patient data NEVER leaves this client.
    Only model weights (floats) are transmitted.
    """
    
    def __init__(self, hospital_id: str):
        self.hospital_id = hospital_id
        self.device = torch.device(cfg.device)
        set_seed(42 + ord(hospital_id))  # Different seed per hospital
        
        print(f"\\n[Hospital {hospital_id}] Initializing FL client...")
        
        # Load this hospital's local dataset split
        self.train_loader, self.val_loader = get_dataloaders(
            hospital_id=hospital_id
        )
        print(f"[Hospital {hospital_id}] "
              f"Train batches: {len(self.train_loader)} | "
              f"Val batches: {len(self.val_loader)}")
        
        # Initialize model
        self.model = self._build_dp_model()
        self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        
        # Privacy tracking
        self.privacy_engine = None
        self.epsilon_spent = 0.0
    
    def _build_dp_model(self):
        """
        Prepare model for Differential Privacy training.
        Opacus requires replacing certain layers (BatchNorm→GroupNorm).
        """
        model = get_model(self.device)
        
        if cfg.dp_enabled:
            # Opacus doesn't support BatchNorm — replace with GroupNorm
            model = ModuleValidator.fix(model)
            errors = ModuleValidator.validate(model, strict=False)
            if errors:
                print(f"[Hospital {self.hospital_id}] "
                      f"DP validation warnings: {errors}")
        
        return model
    
    def _attach_privacy_engine(self, optimizer, data_loader):
        """Attach Opacus PrivacyEngine to model and optimizer."""
        self.privacy_engine = PrivacyEngine(accountant="rdp")
        
        # make_private wraps model, optimizer, and dataloader
        # to automatically add calibrated noise to gradients
        model, optimizer, data_loader = \
            self.privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=optimizer,
                data_loader=data_loader,
                target_epsilon=cfg.dp_target_epsilon,
                target_delta=cfg.dp_target_delta,
                epochs=cfg.fl_local_epochs,
                max_grad_norm=cfg.dp_max_grad_norm,
                # Opacus computes noise_multiplier from target_epsilon
            )
        
        sigma = optimizer.noise_multiplier
        print(f"[Hospital {self.hospital_id}] DP enabled: "
              f"ε={cfg.dp_target_epsilon} | δ={cfg.dp_target_delta} | "
              f"σ={sigma:.3f} | C={cfg.dp_max_grad_norm}")
        
        return model, optimizer, data_loader
    
    def get_parameters(self, config: Dict) -> NDArrays:
        """Return current model weights as list of numpy arrays."""
        return [val.cpu().numpy() 
                for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: NDArrays):
        """Load global model weights from server into local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(np.array(v)) 
                      for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(
        self, 
        parameters: NDArrays, 
        config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Called by FL server to train for one round.
        
        Steps:
        1. Load global model weights
        2. Train locally with DP
        3. Return updated weights
        """
        server_round = config.get("server_round", 1)
        local_epochs = config.get("local_epochs", cfg.fl_local_epochs)
        lr = config.get("learning_rate", cfg.learning_rate)
        
        print(f"\\n[Hospital {self.hospital_id}] "
              f"Round {server_round} — Starting local training...")
        
        # 1. Receive global model
        self.set_parameters(parameters)
        
        # 2. Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr, weight_decay=cfg.weight_decay
        )
        
        # 3. Attach Differential Privacy engine
        if cfg.dp_enabled:
            self.model, optimizer, dp_train_loader = \
                self._attach_privacy_engine(
                    optimizer, self.train_loader
                )
            train_loader = dp_train_loader
        else:
            train_loader = self.train_loader
        
        # 4. Local training loop
        total_loss = 0.0
        for epoch in range(local_epochs):
            epoch_loss = self._train_epoch(
                optimizer, train_loader, epoch, local_epochs
            )
            total_loss += epoch_loss
        
        # 5. Compute privacy budget spent this round
        if cfg.dp_enabled and self.privacy_engine:
            self.epsilon_spent = self.privacy_engine.get_epsilon(
                delta=cfg.dp_target_delta
            )
            print(f"[Hospital {self.hospital_id}] "
                  f"Privacy budget used: ε = {self.epsilon_spent:.4f}")
        
        # 6. Return updated weights + metrics
        return (
            self.get_parameters({}),
            len(self.train_loader.dataset),
            {
                "loss": total_loss / local_epochs,
                "epsilon": self.epsilon_spent,
                "hospital_id": self.hospital_id,
            }
        )
    
    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Called by FL server to evaluate global model 
        on this hospital's local validation set.
        """
        self.set_parameters(parameters)
        
        loss, accuracy, auc = self._evaluate()
        
        print(f"[Hospital {self.hospital_id}] "
              f"Val Acc: {100*accuracy:.2f}% | Loss: {loss:.4f}")
        
        return (
            loss,
            len(self.val_loader.dataset),
            {
                "accuracy": accuracy,
                "loss": loss,
                "epsilon": self.epsilon_spent,
            }
        )
    
    def _train_epoch(self, optimizer, loader, epoch, total_epochs):
        """Single training epoch."""
        self.model.train()
        losses = AverageMeter()
        correct = total = 0
        
        pbar = tqdm(
            loader, 
            desc=f"  [H-{self.hospital_id}] "
                 f"Epoch {epoch+1}/{total_epochs}",
            leave=False
        )
        
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            
            # Opacus automatically clips gradients and adds noise
            # during optimizer.step() when DP is enabled
            optimizer.step()
            
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
            losses.update(loss.item(), len(labels))
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.3f}',
                'acc':  f'{100*correct/total:.1f}%'
            })
        
        return losses.avg
    
    @torch.no_grad()
    def _evaluate(self):
        """Validation — no gradients."""
        self.model.eval()
        losses = AverageMeter()
        all_preds, all_labels, all_probs = [], [], []
        
        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            
            losses.update(loss.item(), len(labels))
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        from sklearn.metrics import roc_auc_score
        accuracy = accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
        
        return losses.avg, accuracy, auc


def start_client(hospital_id: str):
    """Connect to FL server and start training."""
    print(f"[Hospital {hospital_id}] Connecting to {cfg.server_address}")
    
    client = HospitalClient(hospital_id=hospital_id)
    
    fl.client.start_numpy_client(
        server_address=cfg.server_address,
        client=client,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hospital", type=str, required=True,
        choices=["A", "B", "C"],
        help="Hospital identifier (A, B, or C)"
    )
    args = parser.parse_args()
    start_client(args.hospital)
