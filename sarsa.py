import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import time

def train_sarsa(episodes, learning_rate=0.9, discount_factor=0.9, save=True):
    env = gym.make('MountainCar-v0')

    # Discretize the state space
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)

    q_table = np.zeros((len(pos_space), len(vel_space), env.action_space.n))

    epsilon = 1
    epsilon_decay_rate = 2 / episodes
    random = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)
    success_count = 0

    start_time = time.time()

    for i in tqdm(range(episodes)):
        state = env.reset()[0]
        position = np.digitize(state[0], pos_space)
        velocity = np.digitize(state[1], vel_space)

        if random.random() < epsilon:
            action = env.action_space.sample() # explore
        else:
            action = np.argmax(q_table[position, velocity, :]) # exploit

        terminated = False
        rewards = 0

        while not terminated and rewards > -1000:
            new_state, reward, terminated, _, _ = env.step(action)
            new_state_position = np.digitize(new_state[0], pos_space)
            new_state_velocity = np.digitize(new_state[1], vel_space)

            # Choose the next action using epsilon-greedy
            if random.random() < epsilon:
                new_action = env.action_space.sample() # explore
            else:
                new_action = np.argmax(q_table[new_state_position, new_state_velocity, :]) # exploit

            # Update Q-table using the new action
            q_table[position, velocity, action] += learning_rate * (
                reward + discount_factor * q_table[new_state_position, new_state_velocity, new_action] - q_table[position, velocity, action]
            )

            position, velocity, action = new_state_position, new_state_velocity, new_action # move to the next state-action pair
            rewards += reward

        if new_state[0] >= 0.5:
            success_count += 1

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        rewards_per_episode[i] = rewards

    success_rate = success_count / episodes

    end_time = time.time()
    training_time = end_time - start_time

    env.close()

    if save:
        with open('models/mountain_car_sarsa.pkl', 'wb') as f:
            pickle.dump(q_table, f)

    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])

    plt.plot(mean_rewards, label="Mean Reward")
    plt.axhline(0, color='black', linestyle='--', label="Goal Reached")
    plt.title(f"Training Progress: MountainCar (SARSA) - Success Rate: {success_rate*100:.2f}%")
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/mountain_car_sarsa.png')
    plt.show()

    stats = {
        "q_table": q_table,
        "training_time": round(training_time, 2),
    }

    return stats


def evaluate_sarsa(q_table=None, episodes=100, learning_rate=0.9, discount_factor=0.9, render=False):

    env = gym.make('MountainCar-v0', render_mode='human' if render else None)

    # Discretize the state space
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)

    if q_table is None:
        with open('models/mountain_car_sarsa.pkl', 'rb') as f:
            q_table = pickle.load(f)

    rewards_per_episode = np.zeros(episodes)
    success_count = 0

    episode_times = []

    for i in tqdm(range(episodes)):
        state = env.reset()[0]
        position = np.digitize(state[0], pos_space)
        velocity = np.digitize(state[1], vel_space)

        best_action = np.argmax(q_table[position, velocity, :])

        terminated = False
        rewards = 0
        steps = 0

        while not terminated and rewards > -1000:
            new_state, reward, terminated, _, _ = env.step(best_action)
            new_state_position = np.digitize(new_state[0], pos_space)
            new_state_velocity = np.digitize(new_state[1], vel_space)

            new_action = np.argmax(q_table[new_state_position, new_state_velocity, :])

            position, velocity, best_action = new_state_position, new_state_velocity, new_action
            rewards += reward
            steps += 1

        if new_state[0] >= 0.5:
            success_count += 1

        rewards_per_episode[i] = rewards
        episode_times.append(steps)

    success_rate = success_count / episodes

    mean_rewards = np.mean(rewards_per_episode)

    avg_time_to_goal = np.mean(episode_times)

    env.close()

    stats = {
        "success_rate": round(success_rate, 2),
        "rewards_per_episode": [round(r, 2) for r in rewards_per_episode],
        "mean_rewards": round(mean_rewards, 2),
        "average_time": round(avg_time_to_goal, 2),
    }

    return stats
