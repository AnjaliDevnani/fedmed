# config.py
from dataclasses import dataclass, field
from typing import List
import torch

@dataclass
class Config:
    # ── Data ──────────────────────────────────────────
    data_dir: str = "./data/NIH_ChestXray"
    split_dir: str = "./data/hospital_splits"
    image_size: int = 224
    num_classes: int = 2          # Binary: Normal vs Pneumonia
    num_hospitals: int = 3

    # ── Model ─────────────────────────────────────────
    model_name: str = "resnet50"  # or "densenet121", "efficientnet_b0"
    pretrained: bool = True
    dropout_rate: float = 0.3
    
    # ── Training (Central) ────────────────────────────
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 20
    early_stopping_patience: int = 5
    
    # ── Federated Learning ────────────────────────────
    fl_rounds: int = 10
    fl_local_epochs: int = 3
    fl_fraction_fit: float = 1.0   # Use all clients each round
    fl_min_clients: int = 2
    server_address: str = "localhost:8080"
    
    # ── Differential Privacy ──────────────────────────
    dp_enabled: bool = True
    dp_target_epsilon: float = 1.0
    dp_target_delta: float = 1e-5
    dp_max_grad_norm: float = 1.0   # Gradient clipping bound C
    dp_noise_multiplier: float = 1.2 # σ — tune for ε budget
    
    # ── Augmentation ──────────────────────────────────
    use_augmentation: bool = True
    augment_prob: float = 0.5
    
    # ── Hardware ──────────────────────────────────────
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    
    # ── Paths ─────────────────────────────────────────
    import os
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir: str = os.path.join(_base_dir, "checkpoints")
    log_dir: str = os.path.join(_base_dir, "logs")
    results_dir: str = os.path.join(_base_dir, "results")
    global_model_path: str = os.path.join(_base_dir, "checkpoints/global_model.pth")
    
    # ── Class names ───────────────────────────────────
    class_names: List[str] = field(default_factory=lambda: [
        "Normal", "Pneumonia"
    ])
    
    # ── Normalization (ImageNet stats) ────────────────
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std:  List[float]  = field(default_factory=lambda: [0.229, 0.224, 0.225])

cfg = Config()
