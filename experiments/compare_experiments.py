"""
Compare relative losses across Black Scholes, Ornstein-Uhlenbeck, and Heston experiments.
This creates a Figure 2 style plot.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neural_jump_ode.utils import plot_relative_loss
import matplotlib.pyplot as plt
import json


def main():
    """Compare relative losses from all experiments."""
    
    runs_dir = Path("runs")
    
    # Define the experiments to compare
    experiments = [
        ("njode_black_scholes", "Black Scholes"),
        ("njode_ornstein_uhlenbeck", "Ornstein-Uhlenbeck"),
        ("njode_heston", "Heston")
    ]
    
    history_paths = []
    labels = []
    
    # Check which experiments have been run
    for exp_name, label in experiments:
        exp_path = runs_dir / exp_name / "history.json"
        if exp_path.exists():
            history_paths.append(str(exp_path))
            labels.append(label)
            print(f"Found experiment: {label}")
        else:
            print(f"Warning: Experiment {label} not found at {exp_path}")
    
    if not history_paths:
        print("No completed experiments found. Please run the individual experiment scripts first:")
        print("  python experiments/experiment_black_scholes.py")
        print("  python experiments/experiment_ou.py")
        print("  python experiments/experiment_heston.py")
        return
    
    # Create comparison plot
    print(f"\nGenerating comparison plot for {len(history_paths)} experiment(s)...")
    
    plot_relative_loss(
        history_paths=history_paths,
        labels=labels,
        save_path=str(runs_dir / "relative_loss_comparison.png")
    )
    
    # Also create individual summary
    print("\nSummary of final relative losses:")
    for history_path, label in zip(history_paths, labels):
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            if 'relative_loss' in history and history['relative_loss']:
                final_rel_loss = history['relative_loss'][-1]
                print(f"{label:20}: {final_rel_loss:.6f}")
            else:
                print(f"{label:20}: No relative loss data")
        except Exception as e:
            print(f"{label:20}: Error loading data ({e})")
    
    print(f"\nComparison plot saved to: {runs_dir / 'relative_loss_comparison.png'}")


if __name__ == "__main__":
    main()