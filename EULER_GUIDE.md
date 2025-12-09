# Running Neural Jump ODE on ETH Euler Cluster

## Initial Setup (One-time)

1. **SSH to Euler**:
   ```bash
   ssh [your_nethz]@euler.ethz.ch
   ```

2. **Clone repository**:
   ```bash
   cd $HOME  # or your preferred location
   git clone https://github.com/alexander-dybdahl/neural-jump-ode.git
   cd neural-jump-ode
   ```

3. **Run setup script** (creates venv and installs dependencies):
   ```bash
   bash setup_euler.sh
   ```

## Running Experiments

### Option 1: Submit Single Job
Submit a job to SLURM queue:
```bash
sbatch run_heston.sh              # Heston model
sbatch run_black_scholes.sh       # Black-Scholes model
sbatch run_ou.sh                  # Ornstein-Uhlenbeck model
```

### Option 2: Submit with Custom Parameters
Edit the `.sh` file to change command-line arguments, or create a custom script:
```bash
sbatch run_heston.sh
```

### Option 3: Submit Array Job (Hyperparameter Search)
Run multiple configurations in parallel:
```bash
sbatch run_array_job.sh           # Tests 9 combinations of hidden_dim × n_layers
```

### Option 4: GPU Job (for larger experiments)
```bash
sbatch run_gpu.sh                 # Uses GPU with 50k training samples
```

## Monitoring Jobs

**Check job status**:
```bash
squeue -u $USER
```

**View live output**:
```bash
tail -f logs/heston_JOBID.out     # Replace JOBID with actual job ID
```

**Cancel a job**:
```bash
scancel JOBID
```

**View completed job info**:
```bash
sacct -j JOBID --format=JobID,JobName,State,Elapsed,MaxRSS
```

## Adjusting Parameters

You can modify parameters in the `.sh` scripts by editing the command-line arguments:

```bash
python experiments/experiment_heston.py \
    --n-train 20000 \              # More training samples
    --n-val 4000 \                 # More validation samples
    --n-epochs 300 \               # Train longer
    --hidden-dim 100 \             # Larger network
    --n-hidden-layers 2 \          # Deeper network
    --activation tanh \            # Different activation
    --learning-rate 5e-4 \         # Lower learning rate
    --batch-size 256 \             # Larger batch
    --n-steps-between 1 \          # Finer ODE integration
    --device cpu
```

## Resource Recommendations

### CPU Jobs (Default)
- **Small experiments** (n_train=1000):
  - Time: 1 hour
  - CPUs: 4
  - Memory: 4GB per CPU
  
- **Medium experiments** (n_train=10000):
  - Time: 4 hours
  - CPUs: 4
  - Memory: 4GB per CPU
  
- **Large experiments** (n_train=50000):
  - Time: 8 hours
  - CPUs: 8
  - Memory: 8GB per CPU

### GPU Jobs
- Best for: n_train > 20000, hidden_dim > 100
- Time: 2-4 hours
- GPUs: 1
- GPU memory: 8GB
- Speedup: 3-10x faster than CPU

## Optimizations for HPC

The code is already optimized for batch processing:

1. **Cached data generation** (`cache_data=True`):
   - Generates all paths once at start
   - Reuses for all epochs
   - Memory efficient for reasonable n_train (<50k)

2. **Vectorized computations**:
   - Uses PyTorch tensors throughout
   - No Python loops in critical paths

3. **Checkpoint resuming**:
   - Automatically resumes if job is interrupted
   - Saves at each print interval

4. **Parallel experiments**:
   - Use array jobs for hyperparameter search
   - Each job runs independently

## Common Workflow

```bash
# 1. SSH to Euler
ssh [nethz]@euler.ethz.ch

# 2. Navigate to project
cd neural-jump-ode

# 3. Update code (if needed)
git pull

# 4. Submit job
sbatch run_heston.sh

# 5. Check status
squeue -u $USER

# 6. View output
tail -f logs/heston_*.out

# 7. Download results (from local machine)
scp [nethz]@euler.ethz.ch:~/neural-jump-ode/runs/* ./local_runs/
```

## Tips

- **Interactive testing**: Use `bsub -Is bash` for interactive session before submitting jobs
- **Test locally first**: Run with `--n-train 100 --n-epochs 10` to verify code works
- **Monitor memory**: Check `logs/*.err` for out-of-memory errors
- **Use array jobs**: For systematic hyperparameter searches
- **Set cache_data=False**: If memory is limited and you want online learning

## Troubleshooting

**Job fails immediately**:
- Check `logs/*.err` file
- Verify venv exists: `ls venv/`
- Test interactively: `bsub -Is bash`, then run setup

**Out of memory**:
- Reduce `--n-train` or `--batch-size`
- Request more memory: Change `--mem-per-cpu=8192`

**Slow training**:
- Use GPU: `sbatch run_gpu.sh`
- Reduce `--n-steps-between` (default=5, try 1 or 0)
- Increase `--batch-size` if memory allows

**Module not found**:
- Ensure venv is activated in script
- Reinstall: `rm -rf venv && bash setup_euler.sh`
