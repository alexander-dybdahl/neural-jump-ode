"""
Training utilities for Neural Jump ODE models.
"""

import torch
import torch.optim as optim
from typing import List, Dict, Optional, Callable, Tuple
import time
import json
from pathlib import Path

from ..models.jump_ode import NeuralJumpODE, nj_ode_loss


class Trainer:
    def __init__(self, model: NeuralJumpODE, optimizer: optim.Optimizer,
                 device: str = "cpu"):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)
        
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, batch_times: List[torch.Tensor], 
                   batch_values: List[torch.Tensor]) -> float:
        """Train for one epoch."""
        self.model.train()
        
        # Move data to device
        batch_times = [t.to(self.device) for t in batch_times]
        batch_values = [v.to(self.device) for v in batch_values]
        
        self.optimizer.zero_grad()
        
        # Forward pass
        preds, preds_before = self.model(batch_times, batch_values)
        
        # Compute loss
        loss = nj_ode_loss(batch_times, batch_values, preds, preds_before)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, batch_times: List[torch.Tensor], 
                batch_values: List[torch.Tensor]) -> float:
        """Validate model."""
        self.model.eval()
        
        with torch.no_grad():
            # Move data to device
            batch_times = [t.to(self.device) for t in batch_times]
            batch_values = [v.to(self.device) for v in batch_values]
            
            # Forward pass
            preds, preds_before = self.model(batch_times, batch_values)
            
            # Compute loss
            loss = nj_ode_loss(batch_times, batch_values, preds, preds_before)
            
        return loss.item()
    
    def train(self, train_data_fn: Callable, val_data_fn: Optional[Callable] = None,
              n_epochs: int = 100, print_every: int = 10,
              save_path: Optional[str] = None) -> Dict:
        """
        Train the model.
        
        Args:
            train_data_fn: Function that returns (batch_times, batch_values) for training
            val_data_fn: Function that returns validation data
            n_epochs: Number of epochs
            print_every: Print progress every N epochs
            save_path: Path to save the trained model
        """
        
        history = {"train_loss": [], "val_loss": [], "epoch_times": []}
        
        for epoch in range(n_epochs):
            start_time = time.time()
            
            # Get training data
            batch_times, batch_values = train_data_fn()
            
            # Train
            train_loss = self.train_epoch(batch_times, batch_values)
            self.train_losses.append(train_loss)
            history["train_loss"].append(train_loss)
            
            # Validate
            val_loss = None
            if val_data_fn is not None:
                val_batch_times, val_batch_values = val_data_fn()
                val_loss = self.validate(val_batch_times, val_batch_values)
                self.val_losses.append(val_loss)
                history["val_loss"].append(val_loss)
            
            epoch_time = time.time() - start_time
            history["epoch_times"].append(epoch_time)
            
            # Print progress
            if epoch % print_every == 0:
                msg = f"Epoch {epoch:4d} | Train Loss: {train_loss:.6f}"
                if val_loss is not None:
                    msg += f" | Val Loss: {val_loss:.6f}"
                msg += f" | Time: {epoch_time:.2f}s"
                print(msg)
        
        # Save model
        if save_path is not None:
            self.save_model(save_path)
            
        return history
    
    def save_model(self, path: str):
        """Save model state dict."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses
        }, path)
        
    def load_model(self, path: str):
        """Load model state dict."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])


def create_data_loaders(process_type: str = "jump_diffusion", 
                       n_train: int = 100, n_val: int = 20,
                       obs_rate: float = 10.0, **process_kwargs):
    """Create training and validation data generators."""
    
    from ..simulation import create_trajectory_batch
    
    def train_data_fn():
        return create_trajectory_batch(
            n_train, process_type, obs_rate, 
            irregular=True, **process_kwargs
        )
    
    def val_data_fn():
        return create_trajectory_batch(
            n_val, process_type, obs_rate,
            irregular=True, **process_kwargs
        )
    
    return train_data_fn, val_data_fn


def run_experiment(config: Dict, save_dir: str = "runs") -> Dict:
    """
    Run a complete training experiment.
    
    Args:
        config: Dictionary with experiment configuration
        save_dir: Directory to save results
        
    Returns:
        Dictionary with results and metrics
    """
    
    # Create save directory
    save_path = Path(save_dir) / config["experiment_name"]
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(save_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Device selection
    device = config.get("device", "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Clear GPU cache if using CUDA
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create model
    model = NeuralJumpODE(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"],
        dt_between_obs=config.get("dt_between_obs"),
        n_steps_between=config.get("n_steps_between", 0)
    )
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    # Trainer
    trainer = Trainer(model, optimizer, device)
    
    # Data loaders
    train_data_fn, val_data_fn = create_data_loaders(**config["data"])
    
    # Train
    print(f"Starting experiment: {config['experiment_name']}")
    print(f"Device: {device}")
    if device.startswith("cuda"):
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    history = trainer.train(
        train_data_fn=train_data_fn,
        val_data_fn=val_data_fn,
        n_epochs=config["n_epochs"],
        print_every=config.get("print_every", 10),
        save_path=str(save_path / "model.pt")
    )
    
    # Save history
    with open(save_path / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"Experiment completed. Results saved to {save_path}")
    
    return {
        "config": config,
        "history": history,
        "save_path": str(save_path),
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None
    }