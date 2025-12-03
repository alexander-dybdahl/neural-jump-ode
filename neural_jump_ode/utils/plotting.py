"""
Plotting utilities for Neural Jump ODE experiments.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import json
from pathlib import Path

def plot_training_history(history_path: str, save_path: Optional[str] = None):
    """Plot training and validation loss over epochs."""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss', alpha=0.7)
    if history['val_loss']:
        plt.plot(history['val_loss'], label='Validation Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['epoch_times'], alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Training Time per Epoch')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_trajectory_predictions(model, batch_times: List[torch.Tensor], 
                              batch_values: List[torch.Tensor],
                              n_trajectories: int = 3,
                              save_path: Optional[str] = None):
    """
    Plot predicted vs true trajectories with confidence bands.
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Move data to device
    batch_times = [t.to(device) for t in batch_times[:n_trajectories]]
    batch_values = [v.to(device) for v in batch_values[:n_trajectories]]
    
    with torch.no_grad():
        preds, preds_before = model(batch_times, batch_values)
    
    fig, axes = plt.subplots(n_trajectories, 1, figsize=(12, 4*n_trajectories))
    if n_trajectories == 1:
        axes = [axes]
    
    for i in range(n_trajectories):
        times = batch_times[i].cpu().numpy()
        true_values = batch_values[i].cpu().numpy().squeeze()
        pred_values = preds[i].cpu().numpy().squeeze()
        pred_before = preds_before[i].cpu().numpy().squeeze()
        
        ax = axes[i]
        
        # Plot true trajectory
        ax.plot(times, true_values, 'o-', label='True Values', alpha=0.8, markersize=6)
        
        # Plot predictions at observation times
        ax.plot(times, pred_values, 's-', label='Predictions (after jump)', alpha=0.8, markersize=4)
        
        # Plot predictions before jumps
        ax.plot(times, pred_before, '^-', label='Predictions (before jump)', alpha=0.6, markersize=4)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title(f'Trajectory {i+1}: Predictions vs True Values')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_model_predictions_with_confidence(model, test_data_fn, n_samples: int = 100,
                                         n_trajectories: int = 5,
                                         save_path: Optional[str] = None):
    """
    Plot model predictions with confidence bands using multiple samples.
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Generate multiple test batches to compute statistics
    all_predictions = []
    all_true_values = []
    all_times = []
    
    for _ in range(n_samples):
        batch_times, batch_values = test_data_fn()
        batch_times = [t.to(device) for t in batch_times[:n_trajectories]]
        batch_values = [v.to(device) for v in batch_values[:n_trajectories]]
        
        with torch.no_grad():
            preds, _ = model(batch_times, batch_values)
        
        # Store predictions
        for i in range(len(preds)):
            if len(all_predictions) <= i:
                all_predictions.append([])
                all_true_values.append([])
                all_times.append([])
            
            all_predictions[i].append(preds[i].cpu().numpy().squeeze())
            all_true_values[i].append(batch_values[i].cpu().numpy().squeeze())
            all_times[i].append(batch_times[i].cpu().numpy())
    
    # Plot with confidence bands
    fig, axes = plt.subplots(min(n_trajectories, 3), 1, figsize=(12, 4*min(n_trajectories, 3)))
    if min(n_trajectories, 3) == 1:
        axes = [axes]
    
    for i in range(min(n_trajectories, 3)):
        ax = axes[i]
        
        # Get statistics across samples
        pred_array = np.array(all_predictions[i])  # Shape: (n_samples, n_obs)
        true_array = np.array(all_true_values[i])
        times_array = np.array(all_times[i])
        
        # Use the first trajectory's times as reference
        ref_times = times_array[0]
        
        # Compute statistics
        pred_mean = np.mean(pred_array, axis=0)
        pred_std = np.std(pred_array, axis=0)
        true_mean = np.mean(true_array, axis=0)
        
        # Plot confidence bands
        ax.fill_between(ref_times, 
                       pred_mean - 2*pred_std, 
                       pred_mean + 2*pred_std,
                       alpha=0.2, label='95% Confidence Band')
        
        ax.fill_between(ref_times,
                       pred_mean - pred_std,
                       pred_mean + pred_std, 
                       alpha=0.3, label='68% Confidence Band')
        
        # Plot means
        ax.plot(ref_times, pred_mean, '-', linewidth=2, label='Predicted Mean')
        ax.plot(ref_times, true_mean, 'o-', alpha=0.8, label='True Mean')
        
        # Plot sample trajectories
        for j in range(min(5, n_samples)):
            if j == 0:
                ax.plot(times_array[j], pred_array[j], '-', alpha=0.1, color='blue', label='Sample Predictions')
                ax.plot(times_array[j], true_array[j], 'o', alpha=0.3, color='red', markersize=3, label='Sample True Values')
            else:
                ax.plot(times_array[j], pred_array[j], '-', alpha=0.1, color='blue')
                ax.plot(times_array[j], true_array[j], 'o', alpha=0.3, color='red', markersize=3)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title(f'Model Predictions with Confidence Bands (Trajectory Type {i+1})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_loss_components(model, test_data_fn, n_samples: int = 50, 
                        save_path: Optional[str] = None):
    """
    Analyze and plot different components of the loss function.
    """
    from ..models.jump_ode import nj_ode_loss
    
    model.eval()
    device = next(model.parameters()).device
    
    jump_errors = []
    continuous_errors = []
    total_losses = []
    
    for _ in range(n_samples):
        batch_times, batch_values = test_data_fn()
        batch_times = [t.to(device) for t in batch_times]
        batch_values = [v.to(device) for v in batch_values]
        
        with torch.no_grad():
            preds, preds_before = model(batch_times, batch_values)
            
            # Compute loss components
            for x, y, y_before in zip(batch_values, preds, preds_before):
                jump_err = (x - y).abs().pow(2).mean().item()
                cont_err = (y - y_before).abs().pow(2).mean().item()
                total_loss = jump_err + cont_err
                
                jump_errors.append(jump_err)
                continuous_errors.append(cont_err)
                total_losses.append(total_loss)
    
    # Plot loss components
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(jump_errors, bins=20, alpha=0.7, label='Jump Error')
    plt.xlabel('Jump Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Jump Errors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.hist(continuous_errors, bins=20, alpha=0.7, label='Continuous Error', color='orange')
    plt.xlabel('Continuous Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Continuous Errors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.scatter(jump_errors, continuous_errors, alpha=0.6)
    plt.xlabel('Jump Error')
    plt.ylabel('Continuous Error')
    plt.title('Jump vs Continuous Error')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Jump Error - Mean: {np.mean(jump_errors):.6f}, Std: {np.std(jump_errors):.6f}")
    print(f"Continuous Error - Mean: {np.mean(continuous_errors):.6f}, Std: {np.std(continuous_errors):.6f}")
    print(f"Total Loss - Mean: {np.mean(total_losses):.6f}, Std: {np.std(total_losses):.6f}")