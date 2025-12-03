"""
Simple experiment to test Neural Jump ODE on synthetic jump-diffusion data.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neural_jump_ode.utils import run_experiment


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
        "n_epochs": 200,
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


if __name__ == "__main__":
    main()