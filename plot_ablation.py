import numpy as np
import matplotlib.pyplot as plt
from Helper import LearningCurvePlot, smooth

def generate_master_graph(max_episodes=5000):
    plot = LearningCurvePlot(title='Task 2.4')
    plot.set_ylim(0, 520)  # extra space for the max score line
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    linewidth = 2
    alpha = 0.5
    
    experiments = {
        'Naive': 'runs/cartpole_base_noER_noTN_data.npy',
        'Target Network': 'runs/cartpole_base_noER_TN_data.npy',
        'Experience Replay': 'runs/cartpole_base_ER_noTN_data.npy',
        'Full DQN': 'runs/cartpole_base_ER_TN_data.npy'
    }
    
    for i, (label, filepath) in enumerate(experiments.items()):
        try:
            data = np.load(filepath)

            truncated_data = data[:, :max_episodes] if data.shape[1] > max_episodes else data
            mean_rewards = np.mean(truncated_data, axis=0)
            x = np.arange(len(mean_rewards))
            
            window = min(101, len(mean_rewards) - 1 if len(mean_rewards) % 2 == 0 else len(mean_rewards))
            if window >= 3:
                smoothed_mean = smooth(mean_rewards, window=window)
            else:
                smoothed_mean = mean_rewards
            
            plot.ax.plot(x, smoothed_mean,
                         label=label,
                         color=colors[i % len(colors)],
                         linewidth=linewidth,
                         alpha=alpha)
            print(f'Added {label} (truncated to {len(mean_rewards)} episodes).')
        except FileNotFoundError:
            print(f'Skipping {label} - File not found.')
    
    plot.add_hline(500, label='Max Score')
    plot.ax.legend(loc='lower right', framealpha=0.8)
    plot.save('ablation_study_final.png')
    print('\nMaster graph saved to ablation_study_final.png')

if __name__ == '__main__':
    generate_master_graph(max_episodes=5000)