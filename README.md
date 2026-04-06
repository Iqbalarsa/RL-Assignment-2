# Assignment 2: Q-Learning - Tabular & Deep (CartPole)

This repository contains the implementation of a Deep Q-Network (DQN) agent designed to solve the `CartPole-v1` environment. It also includes an automated pipeline for conducting hyperparameter ablation studies.

## Project Structure

* `DQN.py`: The core reinforcement learning agent containing the neural network, replay buffer, and training/testing loops.
* `run_ablation_naive.py`: An automation script to sequentially run the hyperparameter ablation study using the naive Q-learning configuration (no ER, no TN).
* `plot_ablation_naive.py`: Generates visualization curves (Environment Steps vs. Episode Return) for the ablation study.
* `plot_ablation.py`: Generates the master comparison graph evaluating Naive, Only TN, Only ER, and Full DQN configurations against the provided baseline.
* `hyperparameters.yml`: Configuration file containing the hyperparameter dictionaries for all tested variants.
* `Helper.py`: Contains utility functions for smoothing data and plotting. 

## Installation & Setup

1. Ensure you have Python 3.8+ installed.
2. Activate your virtual environment (if using one).
3. Install the required dependencies using the provided requirements file:
```bash
python -m pip install -r requirements.txt
```

## How to Run the Experiments

### 1. Run a Single Configuration (Train or Test)
You can manually trigger the `DQN.py` script to run specific configurations defined in your `hyperparameters.yml` file. 

**To train the Full DQN (with ER and TN):**
```bash
python DQN.py cartpole_base --train --use_er --use_tn
```

**To test a trained model with rendering enabled:**
```bash
python DQN.py cartpole_base --use_er --use_tn
```

### 2. Run the Complete Hyperparameter Ablation Study
To replicate the ablation study for Task 2.2 (evaluating Learning Rate, Update Frequency, Network Size, and Exploration Rate on the naive architecture), run the following command. This will sequentially execute all 9 experimental configurations. 

```bash
python run_ablation_naive.py
```

### 3. Generate Ablation Graphs
Once the ablation study has finished training and the data arrays (`.npy`) are saved in the `runs/` directory, generate the 4 ablation comparison graphs by running:
```bash
python plot_ablation_naive.py
```
This will output `ablation_lr.png`, `ablation_freq.png`, `ablation_net.png`, and `ablation_explore.png` directly into the root directory.

### 4. Generate the Master Comparison Graph (Task 2.4)
To evaluate the impact of Experience Replay and Target Networks, ensure you have trained the four primary baseline configurations:
1. `cartpole_base_noER_noTN`
2. `cartpole_base_noER_TN`
3. `cartpole_base_ER_noTN`
4. `cartpole_base_ER_TN`

Once trained, ensure `BaselineDataCartPole.csv` is in your root directory and run:
```bash
python plot_ablation.py
```
This will read the training data and output the final `master_comparison_steps.png` graph.