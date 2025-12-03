"""
Basic tests for Neural Jump ODE functionality.
"""

import sys
from pathlib import Path
import torch

# Add project root to path  
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neural_jump_ode.models import NeuralJumpODE, nj_ode_loss
from neural_jump_ode.simulation import create_trajectory_batch


def test_model_creation():
    """Test that model can be created and has correct structure."""
    print("Testing model creation...")
    
    model = NeuralJumpODE(
        input_dim=1,
        hidden_dim=16, 
        output_dim=1,
        n_steps_between=3
    )
    
    # Check components exist
    assert hasattr(model, 'jump_nn')
    assert hasattr(model, 'ode_func') 
    assert hasattr(model, 'output_nn')
    
    # Check parameter count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model has {n_params} parameters")
    
    print("✓ Model creation test passed")


def test_forward_pass():
    """Test forward pass with synthetic data."""
    print("Testing forward pass...")
    
    model = NeuralJumpODE(input_dim=1, hidden_dim=16, output_dim=1)
    
    # Create simple test data
    times1 = torch.tensor([0.0, 0.5, 1.0])
    values1 = torch.tensor([[1.0], [1.2], [0.9]]) 
    
    times2 = torch.tensor([0.0, 0.3, 0.8])
    values2 = torch.tensor([[0.5], [0.7], [0.6]])
    
    batch_times = [times1, times2]
    batch_values = [values1, values2]
    
    # Forward pass
    preds, preds_before = model(batch_times, batch_values)
    
    # Check outputs
    assert len(preds) == 2
    assert len(preds_before) == 2
    assert preds[0].shape == (3, 1)
    assert preds[1].shape == (3, 1)
    
    print("  Output shapes are correct")
    print(f"  Pred 1 range: [{preds[0].min():.3f}, {preds[0].max():.3f}]")
    print("✓ Forward pass test passed")


def test_loss_computation():
    """Test loss computation."""
    print("Testing loss computation...")
    
    model = NeuralJumpODE(input_dim=1, hidden_dim=8, output_dim=1)
    
    # Generate synthetic data
    batch_times, batch_values = create_trajectory_batch(
        n_trajectories=3, 
        process_type="jump_diffusion",
        obs_rate=5.0,
        T=0.5,
        jump_rate=1.0
    )
    
    # Forward pass
    preds, preds_before = model(batch_times, batch_values)
    
    # Compute loss
    loss = nj_ode_loss(batch_times, batch_values, preds, preds_before)
    
    # Check loss properties
    assert torch.isfinite(loss), "Loss should be finite"
    assert loss.item() > 0, "Loss should be positive"
    
    print(f"  Loss value: {loss.item():.6f}")
    print("✓ Loss computation test passed")


def test_gradient_flow():
    """Test that gradients can flow through the model."""
    print("Testing gradient flow...")
    
    # Test on GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = NeuralJumpODE(input_dim=1, hidden_dim=8, output_dim=1)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Generate data
    batch_times, batch_values = create_trajectory_batch(
        n_trajectories=2, process_type="ou_jumps", T=0.3, obs_rate=8.0
    )
    
    # Move data to device
    batch_times = [t.to(device) for t in batch_times]
    batch_values = [v.to(device) for v in batch_values]
    
    # Training step
    optimizer.zero_grad()
    preds, preds_before = model(batch_times, batch_values)
    loss = nj_ode_loss(batch_times, batch_values, preds, preds_before)
    loss.backward()
    
    # Check gradients exist
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())
    
    assert len(grad_norms) > 0, "Some gradients should exist"
    assert all(torch.isfinite(torch.tensor(g)) for g in grad_norms), "Gradients should be finite"
    
    optimizer.step()
    
    print(f"  Using device: {device}")
    print(f"  Computed gradients for {len(grad_norms)} parameter groups")
    print(f"  Max gradient norm: {max(grad_norms):.6f}")
    print("✓ Gradient flow test passed")


def test_data_generation():
    """Test synthetic data generation."""
    print("Testing data generation...")
    
    # Test jump-diffusion
    batch_times, batch_values = create_trajectory_batch(
        n_trajectories=5,
        process_type="jump_diffusion", 
        T=1.0,
        obs_rate=10.0
    )
    
    assert len(batch_times) == 5
    assert len(batch_values) == 5
    
    # Check trajectory properties
    for i, (times, values) in enumerate(zip(batch_times, batch_values)):
        assert len(times) == len(values)
        assert times[0] == 0.0  # Should start at t=0
        assert torch.all(times[1:] > times[:-1])  # Should be sorted
        print(f"  Trajectory {i}: {len(times)} observations over [{times[0]:.2f}, {times[-1]:.2f}]")
    
    print("✓ Data generation test passed")


def test_gpu_compatibility():
    """Test GPU compatibility if CUDA is available."""
    if not torch.cuda.is_available():
        print("GPU compatibility test skipped (CUDA not available)")
        return
        
    print("Testing GPU compatibility...")
    device = "cuda"
    
    # Create model on GPU
    model = NeuralJumpODE(input_dim=1, hidden_dim=16, output_dim=1)
    model.to(device)
    
    # Generate data and move to GPU
    batch_times, batch_values = create_trajectory_batch(
        n_trajectories=3, process_type="jump_diffusion", T=0.5, obs_rate=5.0
    )
    batch_times = [t.to(device) for t in batch_times]
    batch_values = [v.to(device) for v in batch_values]
    
    # Forward pass on GPU
    preds, preds_before = model(batch_times, batch_values)
    loss = nj_ode_loss(batch_times, batch_values, preds, preds_before)
    
    # Check tensors are on GPU
    assert loss.device.type == "cuda"
    assert all(p.device.type == "cuda" for p in preds)
    
    print(f"  GPU memory allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    print("✓ GPU compatibility test passed")


def run_all_tests():
    """Run all tests."""
    print("="*50)
    print("RUNNING NEURAL JUMP ODE TESTS")
    print("="*50)
    
    try:
        test_model_creation()
        print()
        
        test_forward_pass()
        print()
        
        test_loss_computation()
        print()
        
        test_gradient_flow()
        print()
        
        test_data_generation()
        print()
        
        test_gpu_compatibility()
        print()
        
        print("="*50)
        print("ALL TESTS PASSED! ✓")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()