import numpy as np
import matplotlib.pyplot as plt
from Helper import LearningCurvePlot, smooth

def generate_master_graph():
    plot = LearningCurvePlot(title='Task 2.4')
    plot.set_ylim(0, 500) 
    
    experiments = {
        'Naive': 'runs/cartpole1_noER_noTN_data.npy',
        'Target Network': 'runs/cartpole1_noER_TN_data.npy',
        'Experience Replay': 'runs/cartpole1_ER_noTN_data.npy',
        'Full DQN': 'runs/cartpole1_ER_TN_data.npy'
    }

    for label, filepath in experiments.items():
        try:
            data = np.load(filepath)
            
            mean_rewards = np.mean(data, axis=0)
            x = np.arange(len(mean_rewards))
            
            smoothed_mean = smooth(mean_rewards, window=101) 
            
            plot.add_curve(x, smoothed_mean, label=label)
                                 
            print(f'Added {label} to the graph.')
            
        except FileNotFoundError:
            print(f'Skipping {label} - File not found.')

    plot.add_hline(500, label='Max Score')
    plot.save('ablation_study_final.png')
    print('\nMaster graph saved to ablation_study_final.png')

if __name__ == '__main__':
    generate_master_graph()