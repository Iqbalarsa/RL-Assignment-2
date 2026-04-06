import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from Helper import smooth

def format_steps(x, pos):
    if x >= 1_000_000:
        return f'{x/1_000_000:.1f}M'
    elif x >= 1_000:
        return f'{int(x/1000)}k'
    else:
        return str(int(x))

def get_mean_and_smooth(rewards, steps, window=501):
    mean_rewards = np.mean(rewards, axis=0)
    mean_steps = np.mean(steps, axis=0)
    if len(mean_rewards) >= window:
        smoothed = smooth(mean_rewards, window=window)
    else:
        smoothed = mean_rewards
    return mean_steps, smoothed

def plot_comparison(window=501, baseline_csv='BaselineDataCartPole.csv'):
    fig, ax = plt.subplots(figsize=(8, 5))

    variants = {
        'Naive': 'cartpole_base_noER_noTN',
        'Target Network': 'cartpole_base_noER_TN',
        'Experience Replay': 'cartpole_base_ER_noTN',
        'Full DQN': 'cartpole_base_ER_TN'
    }

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    linewidth = 2
    alpha = 0.5

    for i, (label, base) in enumerate(variants.items()):
        try:
            rewards = np.load(f'runs/{base}_data.npy')
            steps = np.load(f'runs/{base}_steps.npy')

            n_episodes = min(rewards.shape[1], steps.shape[1])
            if rewards.shape[1] != steps.shape[1]:
                print(f"Warning: {label} episode mismatch. Truncating to {n_episodes}.")
                rewards = rewards[:, :n_episodes]
                steps = steps[:, :n_episodes]

            mean_steps, smoothed = get_mean_and_smooth(rewards, steps, window)

            ax.plot(mean_steps, smoothed,
                    color=colors[i],
                    linewidth=linewidth,
                    alpha=alpha,
                    label=label)
            print(f'Added {label} ({len(mean_steps)} episodes, {mean_steps[-1]:.0f} steps).')

        except FileNotFoundError as e:
            print(f'Skipping {label} – file not found: {e.filename}')

    try:
        df = pd.read_csv(baseline_csv)
        df_grouped = df.groupby('env_step', as_index=False)['Episode_Return_smooth'].mean()
        df_grouped = df_grouped.sort_values('env_step')
    
        ax.plot(df_grouped['env_step'], df_grouped['Episode_Return_smooth'],
            color='purple',      
            linestyle='--',
            linewidth=2,
            alpha=0.3,
            label='Baseline')
    except Exception as e:
        print(f"Baseline error: {e}")

    ax.xaxis.set_major_formatter(FuncFormatter(format_steps))
    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Episode Return')
    ax.set_title('Comparison: Target Network & Experience Replay')
    ax.axhline(500, linestyle=':', color='gray', label='Max Score')
    ax.legend(loc='upper left', framealpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 520)

    plt.tight_layout()
    plt.savefig('master_comparison_steps.png', dpi=300)
    plt.close()
    print("\nMaster graph saved as 'master_comparison_steps.png'")


if __name__ == '__main__':
    # main()
    plot_comparison(window=501)