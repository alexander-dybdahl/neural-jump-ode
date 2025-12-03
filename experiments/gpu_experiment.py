"""
GPU-accelerated experiment for Neural Jump ODE.
Automatically uses CUDA if available, with larger batch sizes and more epochs.
"""

import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neural_jump_ode.utils import run_experiment


def main():
    """Run a larger Jump ODE experiment optimized for GPU."""
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_gpu = device == "cuda"
    
    print(f"Running experiment on: {device}")
    if use_gpu:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Experiment configuration (scaled for GPU if available)
    config = {
        "experiment_name": "gpu_jump_diffusion" if use_gpu else "cpu_jump_diffusion",
        
        # Model parameters (larger for GPU)
        "input_dim": 1,
        "hidden_dim": 64 if use_gpu else 32,
        "output_dim": 1,
        "n_steps_between": 10 if use_gpu else 5,
        
        # Training parameters (more epochs and data for GPU)
        "learning_rate": 0.001,
        "n_epochs": 500 if use_gpu else 200,
        "print_every": 50 if use_gpu else 20,
        "device": device,
        
        # Data generation parameters (larger batches for GPU)
        "data": {
            "process_type": "jump_diffusion",
            "n_train": 200 if use_gpu else 50,
            "n_val": 40 if use_gpu else 10,
            "obs_rate": 12.0,
            
            # Jump-diffusion parameters
            "jump_rate": 2.5,
            "drift": 0.05,
            "vol": 0.25,
            "jump_mean": 0.0,
            "jump_std": 0.15,
            "T": 1.0,
            "x0": 1.0
        }
    }
    
    # Run experiment - save in project runs folder
    print("\\n" + "="*60)
    print("STARTING GPU-OPTIMIZED EXPERIMENT")
    print("="*60)
    
    results = run_experiment(config, save_dir=str(project_root / "runs"))
    
    print("\\n" + "="*60)
    print("EXPERIMENT RESULTS")
    print("="*60)
    print(f"Device used: {device}")
    print(f"Training trajectories: {config['data']['n_train']}")
    print(f"Final training loss: {results['final_train_loss']:.6f}")
    print(f"Final validation loss: {results['final_val_loss']:.6f}")
    print(f"Results saved to: {results['save_path']}")
    
    if use_gpu:
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")


if __name__ == "__main__":
    main()