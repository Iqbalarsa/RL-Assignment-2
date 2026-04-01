import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from Helper import smooth
import os

# Config
hyperparam_groups = {
    'learning_rate': {
        'medium': 'cartpole_base',
        'low': 'cartpole_lr_low',
        'high': 'cartpole_lr_high',
        'title': 'Effect of Learning Rate',
        'filename': 'ablation_lr.png'
    },
    'train_frequency': {
        'medium': 'cartpole_base',
        'low': 'cartpole_freq_low',
        'high': 'cartpole_freq_high',
        'title': 'Effect of Update-to-Data Ratio',
        'filename': 'ablation_freq.png'
    },
    'network_size': {
        'medium': 'cartpole_base',
        'low': 'cartpole_net_small',
        'high': 'cartpole_net_large',
        'title': 'Effect of Network Size',
        'filename': 'ablation_net.png'
    },
    'exploration': {
        'medium': 'cartpole_base',
        'low': 'cartpole_explore_low',
        'high': 'cartpole_explore_high',
        'title': 'Effect of Exploration (epsilon schedule)',
        'filename': 'ablation_explore.png'
    }
}

MODE = 'noER_noTN'

def load_data(hyperparam_set):

    data_file = f'runs/{hyperparam_set}_{MODE}_data.npy'
    steps_file = f'runs/{hyperparam_set}_{MODE}_steps.npy'
    try:
        rewards = np.load(data_file)   
        steps = np.load(steps_file)   
        return rewards, steps
    except FileNotFoundError:
        print(f"Warning: Could not load {data_file} or {steps_file}. Skipping.")
        return None, None

def format_steps(x, pos):

    if x >= 1_000_000:
        return f'{x/1_000_000:.1f}M'
    elif x >= 1_000:
        return f'{int(x/1000)}k'
    else:
        return str(int(x))

def get_mean_and_smooth(rewards, steps, window=101):

    mean_rewards = np.mean(rewards, axis=0)
    mean_steps = np.mean(steps, axis=0)  
    # Smooth rewards
    if len(mean_rewards) >= window:
        smoothed = smooth(mean_rewards, window=window)
    else:
        smoothed = mean_rewards
    return mean_steps, smoothed

def plot_group(group_name, group, window=101):

    data = {}
    for key in ['medium', 'low', 'high']:
        hyperparam_set = group[key]
        rewards, steps = load_data(hyperparam_set)
        if rewards is None:
            print(f"Skipping {key} for {group_name}, data missing.")
            continue
        mean_steps, smoothed = get_mean_and_smooth(rewards, steps, window)
        data[key] = (mean_steps, smoothed)

    if len(data) < 2:
        print(f"Not enough data for {group_name} – skipping.")
        return

    fig, ax = plt.subplots(figsize=(8,5))

    styles = {
        'low': ('blue', 'Low'),
        'medium': ('green', 'Medium'),
        'high': ('red', 'High')
    }
    
    for key in ['low', 'medium', 'high']:
        if key in data:
            steps, rewards = data[key]
            ax.plot(steps, rewards,
                    color=styles[key][0],
                    linewidth=2,
                    label=styles[key][1], alpha=0.5)

    ax.xaxis.set_major_formatter(FuncFormatter(format_steps))

    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Episode Return')
    ax.set_title(group['title'])
    ax.axhline(500, linestyle=':', color='gray', label='Max Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(group['filename'], dpi=300)
    plt.close()
    print(f"Saved {group['filename']}")

def main():
    print("Generating ablation plots...")
    for group_name, group in hyperparam_groups.items():
        plot_group(group_name, group, window=101)
    print("All plots saved.")

if __name__ == '__main__':
    main()