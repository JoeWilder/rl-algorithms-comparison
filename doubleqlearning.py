import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import time

def train_double_qlearning(episodes, learning_rate=0.9, discount_factor=0.9, save=True):
    env = gym.make('MountainCar-v0')

    # Discretize the state space
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)

    # Initialize two q tables
    q_table1 = np.zeros((len(pos_space), len(vel_space), env.action_space.n))
    q_table2 = np.zeros((len(pos_space), len(vel_space), env.action_space.n))

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

        terminated = False
        rewards = 0

        while not terminated and rewards > -1000:
            if random.random() < epsilon:
                action = env.action_space.sample() # explore
            else:
                action = np.argmax(q_table1[position, velocity, :]) # exploit q1

            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)

            # Alternate updates between q1 and q2
            if random.random() < 0.5:
                # Update q1 using q2 to evaluate
                a_star = np.argmax(q_table1[new_state_p, new_state_v, :])
                q_table1[position, velocity, action] += learning_rate * (
                    reward + discount_factor * q_table2[new_state_p, new_state_v, a_star] - q_table1[position, velocity, action]
                )
            else:
                # Update q2 using q1 to evaluate
                a_star = np.argmax(q_table2[new_state_p, new_state_v, :])
                q_table2[position, velocity, action] += learning_rate * (
                    reward + discount_factor * q_table1[new_state_p, new_state_v, a_star] - q_table2[position, velocity, action]
                )

            state = new_state
            position = new_state_p
            velocity = new_state_v
            rewards += reward

        if new_state[0] >= 0.5: # goal reached if position >= 0.5
            success_count += 1

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        rewards_per_episode[i] = rewards

    success_rate = success_count / episodes

    end_time = time.time()
    training_time = end_time - start_time

    env.close()

    if save:
        file = open('models/mountain_car_double_q_learning.pkl', 'wb')
        pickle.dump((q_table1, q_table2), file)
        file.close()

    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])

    plt.plot(mean_rewards, label="Mean Reward")
    plt.axhline(0, color='black', linestyle='--', label="Goal Reached")
    plt.title(f"Training Progress: MountainCar - Success Rate: {success_rate*100:.2f}%")
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/mountain_car_double_q_learning.png')
    plt.show()

    stats = {
        "q_tables": (q_table1, q_table2),
        "training_time": round(training_time, 2),
    }
    return stats


def evaluate_double_qlearning(q_tables=None, episodes=100, render=False):
    env = gym.make('MountainCar-v0', render_mode='human' if render else None)

    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)

    if q_tables is None:
        f = open('models/mountain_car_double_q_learning.pkl', 'rb')
        q1, q2 = pickle.load(f)
        f.close()
    else:
        q1, q2 = q_tables

    rewards_per_episode = np.zeros(episodes)
    success_count = 0
    episode_times = []

    for i in tqdm(range(episodes)):
        state = env.reset()[0]
        position = np.digitize(state[0], pos_space)
        velocity = np.digitize(state[1], vel_space)

        terminated = False
        rewards = 0
        steps = 0

        while not terminated and rewards > -1000:
            action = np.argmax(q1[position, velocity, :] + q2[position, velocity, :])

            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)

            state = new_state
            position = new_state_p
            velocity = new_state_v
            rewards += reward
            steps += 1

        if new_state[0] >= 0.5: # goal reached if position >= 0.5
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