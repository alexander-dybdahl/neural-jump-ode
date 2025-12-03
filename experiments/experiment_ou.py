"""
Ornstein-Uhlenbeck experiment using the new SDE simulation setup.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neural_jump_ode.utils import (
    run_experiment, 
    plot_training_history, 
    plot_single_trajectory_with_condexp,
    plot_relative_loss_single
)
from neural_jump_ode.models import NeuralJumpODE
import torch


def main():
    """Run an Ornstein-Uhlenbeck experiment."""
    
    # Experiment configuration matching the paper
    config = {
        "experiment_name": "njode_ornstein_uhlenbeck",
        "input_dim": 1,
        "hidden_dim": 32,
        "output_dim": 1,
        "n_steps_between": 5,
        "learning_rate": 1e-3,
        "n_epochs": 200,
        "print_every": 10,
        "device": "auto",
        "ignore_first_continuity": True,
        "data": {
            "process_type": "ornstein_uhlenbeck",
            "n_train": 200,
            "n_val": 50,
            "obs_fraction": 0.1,  # About 10% of grid points as observations
            "cache_data": True,  # Cache data for performance
            "theta": 1.0,
            "mu": 0.5,
            "sigma": 0.3,
            "T": 1.0,
            "n_steps": 100,
            "x0": 0.0,
        },
    }
    
    # Run experiment
    results = run_experiment(config, save_dir="runs")
    
    # Create plots
    save_path = Path(results["save_path"])
    
    # Plot training history
    print("\nGenerating training history plot...")
    plot_training_history(
        str(save_path / "history.json"),
        str(save_path / "training_history.png")
    )
    
    # Plot relative loss if available
    print("Generating relative loss plot...")
    try:
        plot_relative_loss_single(
            str(save_path / "history.json"),
            str(save_path / "relative_loss.png")
        )
    except Exception as e:
        print(f"Could not plot relative loss: {e}")
    
    # Load trained model for trajectory plotting
    print("Generating trajectory comparison plot...")
    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = NeuralJumpODE(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"],
        n_steps_between=config.get("n_steps_between", 0)
    ).to(device)
    
    # Load the trained weights
    checkpoint = torch.load(str(save_path / "model.pt"), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Plot single trajectory comparison
    plot_single_trajectory_with_condexp(
        model=model,
        process_type="ornstein_uhlenbeck",
        process_params={
            "theta": config["data"]["theta"],
            "mu": config["data"]["mu"],
            "sigma": config["data"]["sigma"],
            "T": config["data"]["T"],
            "n_steps": config["data"]["n_steps"],
            "x0": config["data"]["x0"],
        },
        obs_fraction=config["data"]["obs_fraction"],
        seed=42,
        save_path=str(save_path / "trajectory_comparison.png")
    )
    
    print(f"\nExperiment completed successfully!")
    print(f"Results saved in: {save_path}")
    print(f"Final training loss: {results['final_train_loss']:.6f}")
    if results['final_val_loss']:
        print(f"Final validation loss: {results['final_val_loss']:.6f}")


if __name__ == "__main__":
    main()