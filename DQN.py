import torch
import gymnasium
from Network import Network
from ReplayBuffer import ReplayBuffer
import itertools
import yaml
import random
from torch import nn
import os
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import argparse

DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory to save run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok = True)

matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DQN:
    
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
        
        self.hyperparameter_set = hyperparameter_set
        
            
        self.learning_rate_a    = hyperparameters['learning_rate_a']        # learning rate (alpha)
        self.discount_factor_g  = hyperparameters['discount_factor_g']      # discount rate (gamma)
        self.network_sync_rate  = hyperparameters['network_sync_rate']      # number of steps the agent takes before syncing the policy and target network
        self.replay_memory_size = hyperparameters['replay_memory_size']     # size of replay memory
        self.mini_batch_size    = hyperparameters['mini_batch_size']        # size of the training data set sampled from the replay memory
        self.epsilon_init       = hyperparameters['epsilon_init']           # 1 = 100% random actions
        self.epsilon_decay      = hyperparameters['epsilon_decay']          # epsilon decay rate
        self.epsilon_min        = hyperparameters['epsilon_min']            # minimum epsilon value
        self.stop_on_reward     = hyperparameters['stop_on_reward']         # stop training after reaching this number of rewards
        self.fc1_nodes          = hyperparameters['fc1_nodes']
        self.env_make_params    = hyperparameters.get('env_make_params',{}) # Get optional environment-specific parameters, default to empty dict
        
        # Neural Network
        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        
        # Path to Run info
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')
        
    def run(self, is_training = True, render = False):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')
                
        
        # Create the environment        
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None, **self.env_make_params)
        
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        
        rewards_per_episode = []
        
        # Create policy network
        policy_dqn = Network(num_states, num_actions, self.fc1_nodes).to(device)
        
        if is_training:
            memory = ReplayBuffer(self.replay_memory_size)
            
            epsilon = self.epsilon_init

            target_dqn = Network(num_states, num_actions, self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            
            # Policy network optimizer
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr = self.learning_rate_a)
            
            epsilon_history = []
            
            step_count = 0
            
            # Track best reward
            best_reward = -9999999
        
        else:
            # Load learned policy
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            
            # Evaluate model
            policy_dqn.eval()
            
        
        # Train episodes-------------------------------------------------------------------    
        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            terminated = False
            episode_reward = 0
            
            while(not terminated and episode_reward < self.stop_on_reward): #maximal reward per eps is 500
                
                if is_training and random.random() < epsilon: #epsilon greedy
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0).squeeze()).argmax()
                
                # Next action:
                #action = env.action_space.sample()

                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item())
                
                # Accumulated reward
                episode_reward += reward
                
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)
                
                if is_training:
                    memory.append((state, action, new_state, reward, terminated))
                    
                    step_count += 1
                
                # Move to new state    
                state = new_state
                
            # Track accumulate rewards
            rewards_per_episode.append(episode_reward)
            
             # Save model when new best reward is obtained.
            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward
                    
             # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time
                    
            
            
            if len(memory)>self.mini_batch_size:
                
                #Sample from memory
                mini_batch = memory.sample(self.mini_batch_size)
                
                self.optimize(mini_batch, policy_dqn, target_dqn)
                
                # Epsilon decay
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                epsilon_history.append(epsilon)
                
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0
                    
            
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        
        states, actions, new_states, rewards, done = zip(*mini_batch)
        
        states = torch.stack(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        new_states = torch.stack(new_states)
        done = torch.tensor(done, dtype=torch.float32).to(device)
        
        # predict Q values by current states
        q_values = policy_dqn(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q
        with torch.no_grad():
            max_next_q = target_dqn(new_states).max(dim=1)[0]
            target_q = rewards + (1-done) * self.discount_factor_g * max_next_q
            
        #Compute loss for the whole minibatch
        loss = self.loss_fn(current_q, target_q)
        
        #Optimize the policy network
        self.optimizer.zero_grad() 
        loss.backward()            # backpropagation
        self.optimizer.step()      # Update network parameter
        
    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-199):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)       
            
            
            
if __name__ == '__main__':
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)
    
    
    # agent = Agent("cartpole1") 
    # agent.run(is_training=True, render=True)
    
    
    
