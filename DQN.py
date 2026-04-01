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
from Helper import LearningCurvePlot, smooth

DATE_FORMAT = '%m-%d %H:%M:%S'

# Directory to save run info
RUNS_DIR = 'runs'
os.makedirs(RUNS_DIR, exist_ok=True)

matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DQN:

    def __init__(self, hyperparameter_set, use_er=True, use_tn=True):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set
        self.use_er = use_er
        self.use_tn = use_tn
        self.mode_str = f"{'ER' if use_er else 'noER'}_{'TN' if use_tn else 'noTN'}"

        self.max_steps = 1_000_000
        self.num_runs = 5  # configurable

        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.discount_factor_g = hyperparameters['discount_factor_g']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_decay_steps = hyperparameters['epsilon_decay_steps']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.fc1_nodes = hyperparameters['fc1_nodes']
        self.train_frequency = hyperparameters['train_frequency']
        self.epsilon_decay_type = hyperparameters.get('epsilon_decay_type', 'linear')
        self.env_make_params = hyperparameters.get('env_make_params', {})

        # Neural net and optimizer will be created per run
        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        # Paths to run info
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}_{self.mode_str}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}_{self.mode_str}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}_{self.mode_str}.png')

    def run(self, is_training=True, render=False):
        """Entry point: either train or test."""
        if is_training:
            self._train(render)
        else:
            self._test(render)

    def _train(self, render):
        """Training loop."""
        start_time = datetime.now()
        log_message = f'{start_time.strftime(DATE_FORMAT)}: Training starting...'
        print(log_message)
        with open(self.LOG_FILE, 'w') as file:
            file.write(log_message + '\n')

        all_runs_reward = []
        all_runs_steps = []
        for run_number in range(self.num_runs):
            print(f'\nStarting run {run_number + 1}/{self.num_runs}.')

            # Create environment
            env = gymnasium.make('CartPole-v1', render_mode='human' if render else None, **self.env_make_params)
            num_states = env.observation_space.shape[0]
            num_actions = env.action_space.n

            # Networks
            policy_dqn = Network(num_states, num_actions, self.fc1_nodes).to(device)
            target_dqn = Network(num_states, num_actions, self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
            memory = ReplayBuffer(self.replay_memory_size)

            global_step = 0
            epsilon = self.epsilon_init
            best_reward = -float('inf')
            episode = 0
            rewards_per_episode = []
            steps_per_episode = []

            while global_step < self.max_steps:
                state, _ = env.reset()
                state = torch.tensor(state, dtype=torch.float, device=device)
                terminated = False
                truncated = False
                episode_reward = 0.0

                while not (terminated or truncated) and global_step < self.max_steps:
                    # Epsilon-greedy action selection
                    if random.random() < epsilon:
                        action = torch.tensor(env.action_space.sample(), device=device)
                    else:
                        with torch.no_grad():
                            action = policy_dqn(state.unsqueeze(0)).argmax()  # add batch dim

                    # Take step
                    new_state, reward, terminated, truncated, info = env.step(action.item())
                    episode_reward += reward

                    new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                    reward_t = torch.tensor(reward, dtype=torch.float, device=device)
                    done_flag = terminated or truncated

                    # Store transition in replay buffer
                    if self.use_er:
                        memory.append((state, action, new_state, reward_t, done_flag))

                    # Train if conditions met
                    if global_step % self.train_frequency == 0:
                        if self.use_er and len(memory) >= self.mini_batch_size:
                            mini_batch = memory.sample(self.mini_batch_size)
                            self._optimize(mini_batch, policy_dqn, target_dqn)
                        elif not self.use_er:
                            batch_of_one = [(state, action, new_state, reward_t, done_flag)]
                            self._optimize(batch_of_one, policy_dqn, target_dqn)

                    global_step += 1

                    # Decay epsilon
                    if self.epsilon_decay_type == 'exponential':
                        epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    else:   # linear decay
                        if global_step < self.epsilon_decay_steps:
                            epsilon = self.epsilon_init - (self.epsilon_init - self.epsilon_min) * (global_step / self.epsilon_decay_steps)
                        else:
                            epsilon = self.epsilon_min

                    # Sync target network
                    if self.use_tn and global_step % self.network_sync_rate == 0:
                        target_dqn.load_state_dict(policy_dqn.state_dict())

                    state = new_state

                # Episode finished
                rewards_per_episode.append(episode_reward)
                steps_per_episode.append(global_step)
                episode += 1

                if episode_reward > best_reward:
                    best_reward = episode_reward
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)

                if episode % 100 == 0:
                    print(f'Run {run_number+1} | Ep {episode} | Steps {global_step}/{self.max_steps} | Epsilon {epsilon:.2f} | Last score: {episode_reward}')

                # early stop if target reward reached
                # if self.stop_on_reward is not None and episode_reward >= self.stop_on_reward:
                #     print(f"Reached target reward {self.stop_on_reward} at episode {episode}.")
                #     break

            all_runs_reward.append(rewards_per_episode)
            all_runs_steps.append(steps_per_episode)
        print(f'{self.num_runs} runs completed.')
        self._save_graph(all_runs_reward, all_runs_steps)

    def _optimize(self, mini_batch, policy_dqn, target_dqn):
        """Take one optimization step on the policy network."""
        states, actions, new_states, rewards, done = zip(*mini_batch)

        # Stack tensors
        states = torch.stack(states).to(device)
        actions = torch.tensor(actions, device=device)
        rewards = torch.tensor(rewards, device=device)
        new_states = torch.stack(new_states).to(device)
        done = torch.tensor(done, dtype=torch.float32, device=device)

        # Current Q values
        q_values = policy_dqn(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values
        with torch.no_grad():
            if self.use_tn:
                max_next_q = target_dqn(new_states).max(dim=1)[0]
            else:
                max_next_q = policy_dqn(new_states).max(dim=1)[0]
            target_q = rewards + (1 - done) * self.discount_factor_g * max_next_q

        # Compute loss and update
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), max_norm=1.0)
        self.optimizer.step()

    def _test(self, render):
        """Evaluate the agent."""
        env = gymnasium.make('CartPole-v1', render_mode='human' if render else None, **self.env_make_params)
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = Network(num_states, num_actions, self.fc1_nodes).to(device)
        policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
        policy_dqn.eval()

        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float, device=device)
        total_reward = 0
        terminated, truncated = False, False

        while not (terminated or truncated):
            with torch.no_grad():
                action = policy_dqn(state.unsqueeze(0)).argmax().item()
            new_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            state = torch.tensor(new_state, dtype=torch.float, device=device)
            if render:
                env.render()

        print(f"Test run total reward: {total_reward}")
        env.close()

    def _save_graph(self, all_runs_reward, all_runs_steps):
        """Plot learning curve and save data (rewards and steps)."""

        max_length = max(max(len(run) for run in all_runs_reward),
                         max(len(run) for run in all_runs_steps))
        padded_rewards = []
        padded_steps = []
        for reward_run, step_run in zip(all_runs_reward, all_runs_steps):
            # pad rewards with last reward
            if len(reward_run) < max_length:
                reward_pad = [reward_run[-1]] * (max_length - len(reward_run))
                padded_rewards.append(reward_run + reward_pad)
            else:
                padded_rewards.append(reward_run)
            # pad steps with last step
            if len(step_run) < max_length:
                step_pad = [step_run[-1]] * (max_length - len(step_run))
                padded_steps.append(step_run + step_pad)
            else:
                padded_steps.append(step_run)

        padded_rewards = np.array(padded_rewards)
        padded_steps = np.array(padded_steps)

        # Compute mean and std for rewards
        mean_rewards = np.mean(padded_rewards, axis=0)
        std_rewards = np.std(padded_rewards, axis=0)
        episodes = np.arange(max_length)

        # Smooth and plot
        smoothed_mean = smooth(mean_rewards, window=101)
        plot = LearningCurvePlot(title=f'Learning Curve: {self.hyperparameter_set} ({self.mode_str})')
        plot.set_ylim(0, 500)
        plot.add_curve(episodes, smoothed_mean, label='Mean Reward')
        plot.save(self.GRAPH_FILE)

        # Save both reward and step data
        reward_file = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}_{self.mode_str}_data.npy')
        np.save(reward_file, padded_rewards)
        step_file = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}_{self.mode_str}_steps.npy')
        np.save(step_file, padded_steps)
        print(f'Reward data saved to {reward_file}')
        print(f'Step data saved to {step_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='Hyperparameter set name from yaml file')
    parser.add_argument('--train', help='Training mode', action='store_true')
    parser.add_argument('--use_er', help='Enable Experience Replay', action='store_true')
    parser.add_argument('--use_tn', help='Enable Target Network', action='store_true')
    args = parser.parse_args()

    dql = DQN(hyperparameter_set=args.hyperparameters,
              use_er=args.use_er,
              use_tn=args.use_tn)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)