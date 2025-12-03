"""
Quick test experiment with plotting to verify visualization works.
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
    """Run a quick test experiment with plotting."""
    
    # Quick test configuration
    config = {
        "experiment_name": "test_plotting",
        
        # Small model for quick testing
        "input_dim": 1,
        "hidden_dim": 16,
        "output_dim": 1,
        "n_steps_between": 3,
        
        # Quick training
        "learning_rate": 0.01,
        "n_epochs": 20,  # Very few epochs for quick test
        "print_every": 5,
        "device": "auto",
        
        # Small dataset
        "data": {
            "process_type": "jump_diffusion",
            "n_train": 10,
            "n_val": 5,
            "obs_rate": 5.0,
            
            # Jump-diffusion parameters
            "jump_rate": 1.0,
            "drift": 0.05,
            "vol": 0.2,
            "jump_mean": 0.0,
            "jump_std": 0.1,
            "T": 0.5,  # Shorter time horizon
            "x0": 1.0
        }
    }
    
    # Run experiment
    print("Running quick test experiment...")
    results = run_experiment(config, save_dir=str(project_root / "runs"))
    
    print(f"\\nTest completed! Loss: {results['final_train_loss']:.6f}")
    
    # Test plotting
    print("\\nTesting plots...")
    save_dir = Path(results['save_path'])
    
    try:
        # Test training history plot
        print("- Training history plot...")
        plot_training_history(
            str(save_dir / "history.json"),
            save_path=str(save_dir / "test_training_history.png")
        )
        print("  Success")
        
        # Load model for prediction plots
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
        
        _, val_data_fn = create_data_loaders(**config["data"])
        
        # Test trajectory predictions
        print("- Trajectory prediction plot...")
        val_batch_times, val_batch_values = val_data_fn()
        plot_trajectory_predictions(
            model, val_batch_times, val_batch_values,
            n_trajectories=2,
            save_path=str(save_dir / "test_trajectories.png")
        )
        print("  Success")
        
        print(f"\nAll test plots saved to: {save_dir}")
        print("Plotting functionality works!")
        
    except Exception as e:
        print(f"Plotting test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()