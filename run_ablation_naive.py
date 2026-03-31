import subprocess
import os
import time
from datetime import datetime

# ========== CONFIGURATION ==========
SCRIPT_NAME = 'DQN.py'          
HYPERPARAM_SETS = [
    'cartpole_base',          # Base (medium all)
    'cartpole_lr_low',        # Low learning rate
    'cartpole_lr_high',       # High learning rate
    'cartpole_freq_low',      # Frequent updates (train_freq=1)
    'cartpole_freq_high',     # Infrequent updates (train_freq=8)
    'cartpole_net_small',     # Small network (32 nodes)
    'cartpole_net_large',     # Large network (128 nodes)
    'cartpole_explore_low',   # Low exploration
    'cartpole_explore_high'   # High exploration
]
USE_ER = False    
USE_TN = False    
# ====================================

def run_experiment(hyperparam_set):
    """Run a single experiment and log output to a file."""
    print(f"\n{'='*60}")
    print(f"Starting experiment: {hyperparam_set} (ER={USE_ER}, TN={USE_TN})")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    cmd = ['python', SCRIPT_NAME, hyperparam_set, '--train']
    if USE_ER:
        cmd.append('--use_er')
    if USE_TN:
        cmd.append('--use_tn')

    # Log file name
    log_dir = 'ablation_logs_naive'  
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{hyperparam_set}.log")

    print(f"Running: {' '.join(cmd)}")
    print(f"Logging to: {log_file}")

    start_time = time.time()
    with open(log_file, 'w') as f:
        process = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - start_time

    if process.returncode == 0:
        print(f"Finished {hyperparam_set} in {elapsed:.2f} seconds.")
    else:
        print(f"ERROR: {hyperparam_set} exited with code {process.returncode}. Check log.")
    print("-"*60)

def main():
    print("Starting ablation study (naive DQN: no ER, no TN)...")
    for hp_set in HYPERPARAM_SETS:
        run_experiment(hp_set)
    print("\nAll experiments completed.")

if __name__ == '__main__':
    main()