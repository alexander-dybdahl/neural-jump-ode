"""
Heston experiment using the new SDE simulation setup.
"""

import sys
import argparse
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Heston Neural Jump ODE Experiment')
    
    # Model architecture
    parser.add_argument('--hidden-dim', type=int, default=50, help='Hidden dimension size')
    parser.add_argument('--n-hidden-layers', type=int, default=1, help='Number of hidden layers')
    parser.add_argument('--activation', type=str, default='relu', 
                        choices=['relu', 'tanh', 'sigmoid', 'elu', 'leaky_relu', 'selu'],
                        help='Activation function')
    parser.add_argument('--n-steps-between', type=int, default=5, help='Euler steps between observations')
    
    # Training parameters
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--n-epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--print-every', type=int, default=5, help='Print frequency')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cpu/cuda)')
    
    # Moment learning
    parser.add_argument('--num-moments', type=int, default=2, help='Number of moments to learn')
    parser.add_argument('--moment-weights', type=float, nargs='+', default=[1.0, 3.0], 
                        help='Weights for each moment loss')
    parser.add_argument('--shared-network', action='store_true', 
                        help='Use single shared network for all moments (default: separate networks)')
    
    # Data parameters
    parser.add_argument('--n-train', type=int, default=1000, help='Number of training trajectories')
    parser.add_argument('--n-val', type=int, default=200, help='Number of validation trajectories')
    parser.add_argument('--obs-fraction', type=float, default=0.1, help='Fraction of points observed')
    parser.add_argument('--mu', type=float, default=0.5, help='Heston drift parameter')
    parser.add_argument('--kappa', type=float, default=2.0, help='Heston mean reversion speed')
    parser.add_argument('--theta', type=float, default=0.04, help='Heston long-term variance')
    parser.add_argument('--xi', type=float, default=0.5, help='Heston volatility of volatility')
    parser.add_argument('--rho', type=float, default=-0.5, help='Heston correlation')
    parser.add_argument('--T', type=float, default=1.0, help='Time horizon')
    parser.add_argument('--n-steps', type=int, default=100, help='Number of time steps')
    parser.add_argument('--x0', type=float, default=1.0, help='Initial stock price')
    parser.add_argument('--v0', type=float, default=0.04, help='Initial variance')
    
    return parser.parse_args()


def main():
    """Run a Heston experiment."""
    args = parse_args()
    
    # Experiment configuration with command-line overrides
    config = {
        "experiment_name": "njode_heston",
        "input_dim": 1,
        "hidden_dim": args.hidden_dim,
        "output_dim": 1,
        "n_hidden_layers": args.n_hidden_layers,
        "activation": args.activation,
        "n_steps_between": args.n_steps_between,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "print_every": args.print_every,
        "device": args.device,
        "ignore_first_continuity": True,
        "num_moments": args.num_moments,
        "moment_weights": args.moment_weights,
        "shared_network": args.shared_network,
        "data": {
            "process_type": "heston",
            "n_train": args.n_train,
            "n_val": args.n_val,
            "obs_fraction": args.obs_fraction,
            "cache_data": True,
            "mu": args.mu,
            "kappa": args.kappa,
            "theta": args.theta,
            "xi": args.xi,
            "rho": args.rho,
            "T": args.T,
            "n_steps": args.n_steps,
            "x0": args.x0,
            "v0": args.v0,
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
        n_steps_between=config.get("n_steps_between", 0),
        num_moments=config.get("num_moments", 1),
        n_hidden_layers=config.get("n_hidden_layers", 1),
        activation=config.get("activation", "relu"),
        shared_network=config.get("shared_network", False)
    ).to(device)
    
    # Load the trained weights
    checkpoint = torch.load(str(save_path / "model.pt"), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Plot single trajectory comparison
    plot_single_trajectory_with_condexp(
        model=model,
        process_type="heston",
        process_params={
            "mu": config["data"]["mu"],
            "kappa": config["data"]["kappa"],
            "theta": config["data"]["theta"],
            "xi": config["data"]["xi"],
            "rho": config["data"]["rho"],
            "T": config["data"]["T"],
            "n_steps": config["data"]["n_steps"],
            "x0": config["data"]["x0"],
            "v0": config["data"]["v0"],
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