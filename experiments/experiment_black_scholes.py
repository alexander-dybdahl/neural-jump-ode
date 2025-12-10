"""
Black Scholes experiment using the new SDE simulation setup.
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
    parser = argparse.ArgumentParser(description='Black Scholes Neural Jump ODE Experiment')
    
    # Model architecture
    parser.add_argument('--hidden-dim', type=int, default=32, help='Hidden dimension size')
    parser.add_argument('--n-hidden-layers', type=int, default=1, help='Number of hidden layers')
    parser.add_argument('--activation', type=str, default='relu', 
                        choices=['relu', 'tanh', 'sigmoid', 'elu', 'leaky_relu', 'selu'],
                        help='Activation function')
    parser.add_argument('--dropout-rate', type=float, default=0.0, help='Dropout rate for regularization')
    parser.add_argument('--input-scaling', type=str, default='identity',
                        choices=['identity', 'tanh', 'sigmoid'],
                        help='Input scaling function for ODE network')
    parser.add_argument('--dt-ode-step', type=float, default=None, help='Fixed time step for ODE integration (if None, single step between observations)')
    
    # Training parameters
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--n-epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--no-shuffle', action='store_true',
                        help='Disable shuffling of trajectories between mini-batches (default: shuffle enabled)')
    parser.add_argument('--print-every', type=int, default=5, help='Print frequency')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cpu/cuda)')
    
    # Moment learning
    parser.add_argument('--num-moments', type=int, default=2, help='Number of moments to learn')
    parser.add_argument('--moment-weights', type=float, nargs='+', default=[1.0, 10.0], 
                        help='Weights for each moment loss')
    parser.add_argument('--shared-network', action='store_true', 
                        help='Use single shared network for all moments (default: separate networks)')
    
    # Data parameters
    parser.add_argument('--cache-data', action='store_true',
                        help='Cache training data (reuse same paths each epoch). Default: False (generate fresh paths)')
    parser.add_argument('--n-train', type=int, default=1000, help='Number of training trajectories')
    parser.add_argument('--n-val', type=int, default=200, help='Number of validation trajectories')
    parser.add_argument('--obs-fraction', type=float, default=0.1, help='Fraction of points observed')
    parser.add_argument('--mu', type=float, default=0.1, help='Black Scholes drift parameter')
    parser.add_argument('--sigma', type=float, default=0.5, help='Black Scholes volatility parameter')
    parser.add_argument('--T', type=float, default=1.0, help='Time horizon')
    parser.add_argument('--n-steps', type=int, default=100, help='Number of time steps')
    parser.add_argument('--x0', type=float, default=1.0, help='Initial value')
    
    return parser.parse_args()


def main():
    """Run a Black Scholes experiment."""
    args = parse_args()
    
    # Experiment configuration with command-line overrides
    config = {
        "experiment_name": "njode_black_scholes",
        "input_dim": 1,
        "hidden_dim": args.hidden_dim,
        "output_dim": 1,
        "n_hidden_layers": args.n_hidden_layers,
        "activation": args.activation,
        "dropout_rate": args.dropout_rate,
        "input_scaling": args.input_scaling,
        "dt_ode_step": args.dt_ode_step,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "shuffle": not args.no_shuffle,
        "print_every": args.print_every,
        "device": args.device,
        "ignore_first_continuity": True,
        "num_moments": args.num_moments,
        "moment_weights": args.moment_weights,
        "shared_network": args.shared_network,
        "data": {
            "process_type": "black_scholes",
            "n_train": args.n_train,
            "n_val": args.n_val,
            "obs_fraction": args.obs_fraction,
            "cache_data": args.cache_data,
            "mu": args.mu,
            "sigma": args.sigma,
            "T": args.T,
            "n_steps": args.n_steps,
            "x0": args.x0,
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
        dt_ode_step=config.get("dt_ode_step", None),
        num_moments=config.get("num_moments", 1),
        n_hidden_layers=config.get("n_hidden_layers", 1),
        activation=config.get("activation", "relu"),
        shared_network=config.get("shared_network", False),
        dropout_rate=config.get("dropout_rate", 0.0),
        input_scaling=config.get("input_scaling", "identity")
    ).to(device)
    
    # Load the trained weights
    checkpoint = torch.load(str(save_path / "model.pt"), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Plot single trajectory comparison
    plot_single_trajectory_with_condexp(
        model=model,
        process_type="black_scholes",
        process_params={
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