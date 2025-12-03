"""
Test checkpoint resuming functionality.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neural_jump_ode.utils import run_experiment


def main():
    """Test checkpoint resuming by running a short experiment twice."""
    
    config = {
        "experiment_name": "checkpoint_test",
        
        # Model parameters
        "input_dim": 1,
        "hidden_dim": 16,
        "output_dim": 1,
        "n_steps_between": 3,
        
        # Training parameters
        "learning_rate": 0.01,
        "n_epochs": 30,  # Run for 30 epochs total
        "print_every": 5,
        "device": "auto",
        
        # Data parameters
        "data": {
            "process_type": "jump_diffusion",
            "n_train": 20,
            "n_val": 5,
            "obs_rate": 5.0,
            "jump_rate": 1.0,
            "drift": 0.05,
            "vol": 0.2,
            "jump_mean": 0.0,
            "jump_std": 0.1,
            "T": 0.5,
            "x0": 1.0
        }
    }
    
    print("Testing checkpoint resuming functionality...")
    print("="*50)
    
    # First run - train for some epochs
    print("1. First run (fresh training)...")
    config_first = config.copy()
    config_first["n_epochs"] = 15  # Train for first 15 epochs
    
    results1 = run_experiment(config_first, save_dir=str(project_root / "runs"))
    
    print(f"\\nFirst run completed - Loss: {results1['final_train_loss']:.6f}")
    
    # Second run - should resume from checkpoint
    print("\\n2. Second run (should resume from checkpoint)...")
    config_second = config.copy() 
    config_second["n_epochs"] = 30  # Train to epoch 30 (should resume from 15)
    
    results2 = run_experiment(config_second, save_dir=str(project_root / "runs"))
    
    print(f"\\nSecond run completed - Final loss: {results2['final_train_loss']:.6f}")
    
    # Third run - should detect already completed training
    print("\\n3. Third run (should detect completed training)...")
    results3 = run_experiment(config, save_dir=str(project_root / "runs"))
    
    print("\\nCheckpoint resuming test completed!")
    print(f"Final training length: {len(results3['history']['train_loss'])} epochs")


if __name__ == "__main__":
    main()