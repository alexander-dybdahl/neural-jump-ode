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
                 device: str = "cpu", ignore_first_continuity: bool = False, 
                 moment_weights: Optional[List[float]] = None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.ignore_first_continuity = ignore_first_continuity
        self.moment_weights = torch.tensor(moment_weights, device=device) if moment_weights else None
        self.model.to(device)
        
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, batch_times: List[torch.Tensor], 
                   batch_values: List[torch.Tensor], batch_size: Optional[int] = None,
                   shuffle: bool = True) -> float:
        """
        Train for one epoch with mini-batching.
        
        Args:
            batch_times: List of observation time tensors for each trajectory
            batch_values: List of observation value tensors for each trajectory
            batch_size: Number of trajectories per mini-batch. If None, use all data at once.
            shuffle: Whether to shuffle trajectories before creating mini-batches
        
        Returns:
            Average loss across all mini-batches
        """
        self.model.train()
        
        n_trajectories = len(batch_times)
        
        # Create indices for trajectories
        indices = list(range(n_trajectories))
        if shuffle:
            import random
            random.shuffle(indices)
        
        # If no batch_size specified, process all at once
        if batch_size is None or batch_size >= n_trajectories:
            batch_times = [batch_times[i].to(self.device) for i in indices]
            batch_values = [batch_values[i].to(self.device) for i in indices]
            
            self.optimizer.zero_grad()
            preds, preds_before = self.model(batch_times, batch_values)
            loss = nj_ode_loss(batch_times, batch_values, preds, preds_before, 
                             ignore_first_continuity=self.ignore_first_continuity, 
                             moment_weights=self.moment_weights)
            loss.backward()
            self.optimizer.step()
            
            return loss.item()
        
        # Mini-batch training
        total_loss = 0.0
        n_batches = 0
        
        for start_idx in range(0, n_trajectories, batch_size):
            end_idx = min(start_idx + batch_size, n_trajectories)
            mini_batch_indices = indices[start_idx:end_idx]
            
            # Extract mini-batch and move to device
            mini_batch_times = [batch_times[i].to(self.device) for i in mini_batch_indices]
            mini_batch_values = [batch_values[i].to(self.device) for i in mini_batch_indices]
            
            # Forward pass
            self.optimizer.zero_grad()
            preds, preds_before = self.model(mini_batch_times, mini_batch_values)
            
            # Compute loss
            loss = nj_ode_loss(mini_batch_times, mini_batch_values, preds, preds_before,
                             ignore_first_continuity=self.ignore_first_continuity, 
                             moment_weights=self.moment_weights)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
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
            loss = nj_ode_loss(batch_times, batch_values, preds, preds_before, ignore_first_continuity=self.ignore_first_continuity, moment_weights=self.moment_weights)
            
        return loss.item()
    
    def train(self, train_data_fn: Callable, val_data_fn: Optional[Callable] = None,
              n_epochs: int = 100, batch_size: Optional[int] = None, 
              shuffle: bool = True, print_every: int = 10,
              save_path: Optional[str] = None, resume_from_checkpoint: bool = True,
              config: Optional[Dict] = None) -> Dict:
        """
        Train the model.
        
        Args:
            train_data_fn: Function that returns (batch_times, batch_values) for training
            val_data_fn: Function that returns validation data
            n_epochs: Number of epochs
            batch_size: Number of trajectories per mini-batch. If None, use all data at once.
            shuffle: Whether to shuffle trajectories before creating mini-batches
            print_every: Print progress every N epochs
            save_path: Path to save the trained model
            resume_from_checkpoint: If True, resume from existing checkpoint if available
            config: Optional config dict for relative loss computation
        """
        
        start_epoch = 0
        
        # Check for existing checkpoint
        if resume_from_checkpoint and save_path and Path(save_path).exists():
            print(f"Found existing checkpoint at {save_path}")
            try:
                checkpoint = torch.load(save_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.train_losses = checkpoint.get("train_losses", [])
                self.val_losses = checkpoint.get("val_losses", [])
                start_epoch = len(self.train_losses)
                print(f"Resuming from epoch {start_epoch} (previous best loss: {min(self.train_losses):.6f})")
                
                # If already completed training, return existing results
                if start_epoch >= n_epochs:
                    print(f"Training already completed ({start_epoch} >= {n_epochs} epochs)")
                    return {
                        "train_loss": self.train_losses,
                        "val_loss": self.val_losses,
                        "epoch_times": checkpoint.get("epoch_times", []),
                        "relative_loss": checkpoint.get("relative_loss", []),
                        "resumed_from_checkpoint": True
                    }
            except Exception as e:
                print(f"Warning: Could not load checkpoint ({e}). Starting fresh training.")
                start_epoch = 0
        
        history = {
            "train_loss": self.train_losses.copy(), 
            "val_loss": self.val_losses.copy(), 
            "epoch_times": [],
            "relative_loss": []
        }
        
        # Prepare for relative loss computation if config is provided
        compute_relative_loss = config and "data" in config and "process_type" in config["data"]
        if compute_relative_loss:
            process_type = config["data"]["process_type"]
            if process_type in ["black_scholes", "ornstein_uhlenbeck", "heston"]:
                # Import conditional expectation functions
                from ..simulation.data_generation import (
                    bs_condexp_at_obs, ou_condexp_at_obs, heston_condexp_at_obs
                )
                
                # Get a fixed batch for relative loss computation
                eval_batch_times, eval_batch_values = train_data_fn()
                eval_batch_times = [t.to(self.device) for t in eval_batch_times[:10]]  # Use subset for efficiency
                eval_batch_values = [v.to(self.device) for v in eval_batch_values[:10]]
        
        for epoch in range(start_epoch, n_epochs):
            start_time = time.time()
            
            # Get training data
            batch_times, batch_values = train_data_fn()
            
            # Train with mini-batching
            train_loss = self.train_epoch(batch_times, batch_values, 
                                         batch_size=batch_size, shuffle=shuffle)
            self.train_losses.append(train_loss)
            history["train_loss"].append(train_loss)
            
            # Validate
            val_loss = None
            if val_data_fn is not None:
                val_batch_times, val_batch_values = val_data_fn()
                val_loss = self.validate(val_batch_times, val_batch_values)
                self.val_losses.append(val_loss)
                history["val_loss"].append(val_loss)
            
            # Compute relative loss less frequently to speed up training
            if compute_relative_loss and epoch % print_every == 0:
                try:
                    self.model.eval()
                    with torch.no_grad():
                        # Model predictions
                        preds, preds_before = self.model(eval_batch_times, eval_batch_values)
                        L_model = nj_ode_loss(eval_batch_times, eval_batch_values, preds, preds_before, moment_weights=self.moment_weights).item()
                        
                        # True conditional expectations for multiple moments
                        from ..simulation.data_generation import get_conditional_moments_at_obs
                        
                        num_moments = getattr(self.model, 'num_moments', 1)
                        
                        # Extract process parameters without process_type to avoid duplicate argument
                        process_params = {k: v for k, v in config["data"].items() if k != "process_type"}
                        
                        y_true, y_true_before = get_conditional_moments_at_obs(
                            [t.cpu() for t in eval_batch_times], 
                            [v.cpu() for v in eval_batch_values],
                            process_type=process_type,
                            num_moments=num_moments,
                            **process_params
                        )
                        
                        # Move true values to device
                        y_true = [y.to(self.device) for y in y_true]
                        y_true_before = [y.to(self.device) for y in y_true_before]
                        
                        L_true = nj_ode_loss(eval_batch_times, eval_batch_values, y_true, y_true_before, moment_weights=self.moment_weights).item()
                        
                        relative_loss = (L_model - L_true) / max(L_true, 1e-8)  # Avoid division by zero
                        history["relative_loss"].append(relative_loss)
                        
                except Exception as e:
                    print(f"Warning: Could not compute relative loss at epoch {epoch}: {e}")
                    history["relative_loss"].append(float('nan'))
            
            epoch_time = time.time() - start_time
            history["epoch_times"].append(epoch_time)
            
            # Print progress and save model
            if epoch % print_every == 0 or epoch == start_epoch:
                msg = f"Epoch {epoch:4d} | Train Loss: {train_loss:.6f}"
                if val_loss is not None:
                    msg += f" | Val Loss: {val_loss:.6f}"
                if history["relative_loss"]:
                    msg += f" | Rel Loss: {history['relative_loss'][-1]:.4f}"
                msg += f" | Time: {epoch_time:.2f}s"
                if start_epoch > 0 and epoch == start_epoch:
                    msg += " (resumed)"
                print(msg)
                
                # Save model checkpoint each time we print
                if save_path is not None:
                    self.save_model(save_path, history["epoch_times"], history["relative_loss"])
        
        # Save model
        if save_path is not None:
            self.save_model(save_path, history["epoch_times"], history["relative_loss"])
            
        return history
    
    def save_model(self, path: str, epoch_times: Optional[List[float]] = None, 
                   relative_loss: Optional[List[float]] = None):
        """Save model state dict."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "epoch_times": epoch_times or [],
            "relative_loss": relative_loss or []
        }, path)
        
    def load_model(self, path: str):
        """Load model state dict."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])


def create_data_loaders(process_type: str = "black_scholes", 
                       n_train: int = 100, n_val: int = 20,
                       obs_fraction: float = 0.1,
                       cache_data: bool = True,
                       **process_kwargs):
    """Create training and validation data generators with optional caching."""
    
    from ..simulation import create_trajectory_batch
    
    if cache_data:
        # Generate data once and cache it
        train_data = create_trajectory_batch(
            n_train, process_type, obs_fraction=obs_fraction, **process_kwargs
        )
        val_data = create_trajectory_batch(
            n_val, process_type, obs_fraction=obs_fraction, **process_kwargs
        )
        
        def train_data_fn():
            return train_data
        
        def val_data_fn():
            return val_data
    else:
        # Generate data fresh each time (old behavior)
        def train_data_fn():
            return create_trajectory_batch(
                n_train, process_type, obs_fraction=obs_fraction, **process_kwargs
            )
        
        def val_data_fn():
            return create_trajectory_batch(
                n_val, process_type, obs_fraction=obs_fraction, **process_kwargs
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
        n_steps_between=config.get("n_steps_between", 0),
        num_moments=config.get("num_moments", 1),
        n_hidden_layers=config.get("n_hidden_layers", 1),
        activation=config.get("activation", "relu"),
        shared_network=config.get("shared_network", False),
        dropout_rate=config.get("dropout_rate", 0.1),
        input_scaling=config.get("input_scaling", "identity")
    )
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    
    # Trainer
    trainer = Trainer(model, optimizer, device, ignore_first_continuity=config.get("ignore_first_continuity", False), moment_weights=config.get("moment_weights"))
    
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
        batch_size=config.get("batch_size"),
        shuffle=config.get("shuffle", True),
        print_every=config.get("print_every", 10),
        save_path=str(save_path / "model.pt"),
        resume_from_checkpoint=config.get("resume_from_checkpoint", True),
        config=config
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