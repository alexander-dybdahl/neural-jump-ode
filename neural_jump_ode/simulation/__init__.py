"""Simulation package for generating synthetic time series data"""

from .data_generation import (
    generate_jump_diffusion,
    generate_ou_with_jumps,
    create_trajectory_batch,
    subsample_trajectory
)

__all__ = [
    "generate_jump_diffusion",
    "generate_ou_with_jumps", 
    "create_trajectory_batch",
    "subsample_trajectory"
]
