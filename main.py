from qlearning import train_qlearning, evaluate_qlearning
from doubleqlearning import train_double_qlearning, evaluate_double_qlearning
from sarsa import train_sarsa, evaluate_sarsa
from dqn import train_dqn, evaluate_dqn, DQN
import numpy as np
import torch
import gymnasium as gym

TRAIN_MODE = False

if __name__ == '__main__':
    
    episodes = 1000
    train_results = train_qlearning(episodes)
    q_table = train_results["q_table"]
    results = evaluate_qlearning(q_table, render=True)

    print(f"---------- Q-Learning Metrics ----------")
    print(f"Episodes used to train: {episodes} episodes")
    print(f"Time required to train: {train_results['training_time']} seconds")
    print(f"Evaluation success rate: {results['success_rate']}")
    print(f"Policy variance: {round(np.var(results['rewards_per_episode']), 2)}")
    print(f"Mean reward: {results['mean_rewards']}")
    print(f"Average completion time: {results['average_time']} timesteps")
    print(f"Plot location: plots/mountain_car_q_learning.png\n\n\n")

    ####################################################################

    train_results = train_double_qlearning(episodes)
    q_table1, q_table2 = train_results["q_tables"]
    results = evaluate_double_qlearning(train_results["q_tables"])

    print(f"---------- Double Q-Learning Metrics ----------")
    print(f"Episodes used to train: {episodes} episodes")
    print(f"Time required to train: {train_results['training_time']} seconds")
    print(f"Evaluation success rate: {results['success_rate']}")
    print(f"Policy variance: {round(np.var(results['rewards_per_episode']), 2)}")
    print(f"Mean reward: {results['mean_rewards']}")
    print(f"Average completion time: {results['average_time']} timesteps")
    print(f"Plot location: plots/mountain_car_double_q_learning.png\n\n\n")

    ####################################################################
    
    train_results = train_sarsa(episodes)
    results = evaluate_sarsa(train_results["q_table"])

    print(f"---------- SARSA Metrics ----------")
    print(f"Episodes used to train: {episodes} episodes")
    print(f"Time required to train: {train_results['training_time']} seconds")
    print(f"Evaluation success rate: {results['success_rate']}")
    print(f"Policy variance: {round(np.var(results['rewards_per_episode']), 2)}")
    print(f"Mean reward: {results['mean_rewards']}")
    print(f"Average completion time: {results['average_time']} timesteps")
    print(f"Plot location: plots/mountain_car_sarsa.png\n\n\n")

    ####################################################################

    training_time = 2996.88
    env = gym.make("MountainCar-v0")
    agent = DQN(env=env)
    agent.load("models/mountain_car_dqn")
    results = evaluate_dqn(agent)
    
    print(f"---------- DQN Metrics ----------")
    print(f"Episodes used to train: {episodes} episodes")
    print(f"Time required to train: {training_time} seconds")
    print(f"Evaluation success rate: {results['success_rate']}")
    print(f"Policy variance: {round(np.var(results['rewards_per_episode']), 2)}")
    print(f"Mean reward: {results['mean_rewards']}")
    print(f"Average completion time: {results['average_time']} timesteps")
    print(f"Plot location: plots/mountain_car_dqn.png\n\n\n")
    
