"""
Simple experiment to test Neural Jump ODE on synthetic jump-diffusion data.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neural_jump_ode.utils import (
    run_experiment, create_data_loaders,
    plot_training_history, plot_trajectory_predictions,
    plot_model_predictions_with_confidence, plot_loss_components
)
from neural_jump_ode.models import NeuralJumpODE
import torch


def main():
    """Run a basic Jump ODE experiment on synthetic data."""
    
    # Experiment configuration
    config = {
        "experiment_name": "basic_jump_diffusion",
        
        # Model parameters
        "input_dim": 1,
        "hidden_dim": 32,
        "output_dim": 1,
        "n_steps_between": 5,  # Intermediate ODE steps
        
        # Training parameters
        "learning_rate": 0.001,
        "n_epochs": 1000,
        "print_every": 20,
        "device": "auto",
        
        # Data generation parameters
        "data": {
            "process_type": "jump_diffusion",
            "n_train": 50,
            "n_val": 10,
            "obs_rate": 8.0,  # observations per unit time
            
            # Jump-diffusion parameters
            "jump_rate": 2.0,
            "drift": 0.05,
            "vol": 0.2,
            "jump_mean": 0.0,
            "jump_std": 0.1,
            "T": 1.0,
            "x0": 1.0
        }
    }
    
    # Run experiment
    results = run_experiment(config, save_dir=str(project_root / "runs"))
    
    print("\n" + "="*50)
    print("EXPERIMENT RESULTS")
    print("="*50)
    print(f"Final training loss: {results['final_train_loss']:.6f}")
    print(f"Final validation loss: {results['final_val_loss']:.6f}")
    print(f"Results saved to: {results['save_path']}")
    
    # Create plots
    print("\nGenerating plots...")
    save_dir = Path(results['save_path'])
    
    # Plot training history
    print("1. Training history...")
    plot_training_history(
        str(save_dir / "history.json"),
        save_path=str(save_dir / "training_history.png")
    )
    
    # Load the trained model for predictions
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NeuralJumpODE(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"], 
        output_dim=config["output_dim"],
        n_steps_between=config.get("n_steps_between", 0)
    )
    checkpoint = torch.load(save_dir / "model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    
    # Create data loaders for testing
    _, val_data_fn = create_data_loaders(**config["data"])
    
    # Plot trajectory predictions
    print("2. Trajectory predictions...")
    val_batch_times, val_batch_values = val_data_fn()
    plot_trajectory_predictions(
        model, val_batch_times, val_batch_values,
        n_trajectories=3,
        save_path=str(save_dir / "trajectory_predictions.png")
    )
    
    # Plot predictions with confidence bands
    print("3. Confidence bands...")
    plot_model_predictions_with_confidence(
        model, val_data_fn, n_samples=50, n_trajectories=3,
        save_path=str(save_dir / "confidence_predictions.png")
    )
    
    # Plot loss components
    print("4. Loss analysis...")
    plot_loss_components(
        model, val_data_fn, n_samples=30,
        save_path=str(save_dir / "loss_components.png")
    )
    
    print(f"\nAll plots saved to: {save_dir}")


if __name__ == "__main__":
    main()