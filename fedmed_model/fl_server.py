# fl_server.py
"""
Flower Federated Learning Server.
Implements FedAvg aggregation with custom strategy.
Runs on cloud (AWS EKS / local for testing).
"""
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
from typing import List, Tuple, Optional, Dict
import numpy as np
import json
import os
from datetime import datetime

from config import cfg
from model import get_model
from utils import save_checkpoint

# ── Track training history ──────────────────────────────
training_history = {
    'rounds': [], 'accuracy': [], 'loss': [], 
    'privacy_epsilon': []
}


def weighted_average_metrics(
    metrics: List[Tuple[int, Metrics]]
) -> Metrics:
    """
    Aggregate evaluation metrics from all clients.
    Weight each client's metric by its dataset size.
    This is standard FedAvg metric aggregation.
    """
    accuracies = [num * m["accuracy"] for num, m in metrics]
    losses     = [num * m["loss"]     for num, m in metrics]
    total      = sum(num for num, _ in metrics)
    
    # Collect epsilon values (privacy budget) from clients
    epsilons = [m.get("epsilon", 0.0) for _, m in metrics]
    
    return {
        "accuracy": sum(accuracies) / total,
        "loss":     sum(losses)     / total,
        "epsilon":  max(epsilons),   # Worst-case privacy budget
    }


def on_fit_config(server_round: int) -> Dict:
    """
    Send training configuration to clients each round.
    Can adapt hyperparameters based on round number.
    """
    return {
        "server_round": server_round,
        "local_epochs": cfg.fl_local_epochs,
        "batch_size":   cfg.batch_size,
        # Reduce LR in later rounds for fine-tuning
        "learning_rate": cfg.learning_rate * (0.9 ** server_round),
    }


def on_evaluate_config(server_round: int) -> Dict:
    """Config sent to clients for evaluation rounds."""
    return {"server_round": server_round, "val_steps": 50}


class FedMedStrategy(FedAvg):
    """
    Custom FedAvg strategy with:
    - Logging of each round's metrics
    - Model saving after each round
    - Privacy budget tracking
    """
    def aggregate_fit(self, server_round, results, failures):
        """Called after clients send their updated weights."""
        print(f"\\n[Server] Round {server_round} — "
              f"Aggregating {len(results)} clients")
        
        # Standard FedAvg aggregation
        aggregated = super().aggregate_fit(
            server_round, results, failures
        )
        
        if aggregated is not None:
            weights, _ = aggregated
            # Save global model after each round
            self._save_global_model(weights, server_round)
        
        return aggregated
    
    def aggregate_evaluate(self, server_round, results, failures):
        """Called after clients send their evaluation metrics."""
        aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )
        
        if aggregated is not None:
            loss, metrics = aggregated
            print(f"[Server] Round {server_round} — "
                  f"Global Acc: {100*metrics['accuracy']:.2f}% | "
                  f"Loss: {loss:.4f} | "
                  f"ε: {metrics.get('epsilon', 'N/A')}")
            
            # Log to history
            training_history['rounds'].append(server_round)
            training_history['accuracy'].append(metrics['accuracy'])
            training_history['loss'].append(loss)
            training_history['privacy_epsilon'].append(
                metrics.get('epsilon', 0.0)
            )
            
            # Save history to JSON for dashboard
            os.makedirs(cfg.results_dir, exist_ok=True)
            with open(f"{cfg.results_dir}/fl_history.json", 'w') as f:
                json.dump(training_history, f, indent=2)
        
        return aggregated
    
    def _save_global_model(self, parameters, server_round: int):
        """Convert Flower parameters to PyTorch and save."""
        import torch
        model = get_model()
        # Convert numpy arrays to state dict
        params_dict = zip(model.state_dict().keys(), parameters.tensors)
        state_dict = {k: torch.tensor(np.array(v)) 
                      for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        
        path = f"{cfg.checkpoint_dir}/global_round_{server_round}.pth"
        save_checkpoint({'model_state_dict': model.state_dict(),
                        'round': server_round}, path)


def start_server():
    strategy = FedMedStrategy(
        fraction_fit=cfg.fl_fraction_fit,
        fraction_evaluate=cfg.fl_fraction_fit,
        min_fit_clients=cfg.fl_min_clients,
        min_evaluate_clients=cfg.fl_min_clients,
        min_available_clients=cfg.fl_min_clients,
        on_fit_config_fn=on_fit_config,
        on_evaluate_config_fn=on_evaluate_config,
        evaluate_metrics_aggregation_fn=weighted_average_metrics,
    )
    
    print(f"[FL Server] Starting on {cfg.server_address}")
    print(f"[FL Server] Rounds: {cfg.fl_rounds} | "
          f"Min clients: {cfg.fl_min_clients}")
    
    fl.server.start_server(
        server_address=cfg.server_address,
        config=fl.server.ServerConfig(num_rounds=cfg.fl_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    start_server()
