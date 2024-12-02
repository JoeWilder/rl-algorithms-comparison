import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import random
from collections import deque
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

class DQN:
    def __init__(self, env):
        self.env = env

        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.batch_size = 128
        self.gamma = 0.99
        self.tau = 0.005 # target network update rate
        self.replayBuffer = deque(maxlen=100000) # store previous experiences

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.build_network().to(self.device)
        self.targetModel = self.build_network().to(self.device)

        self.update_weights()

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def build_network(self):
        input_shape = self.env.observation_space.shape[0]
        output_shape = self.env.action_space.n

        class DQNNetwork(nn.Module):
            def __init__(self, input_shape, output_shape):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Linear(input_shape, 64),
                    nn.ReLU(),
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 64),
                    nn.ReLU()
                )
                self.output = nn.Linear(64, output_shape)

            def forward(self, x):
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                features = self.features(x)
                return self.output(features)

        return DQNNetwork(input_shape, output_shape)

    def predict_action(self, state, evaluate=False):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        
        if not evaluate:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if np.random.rand() < self.epsilon and not evaluate:
            action = random.randint(0, self.env.action_space.n - 1)
        else:
            with torch.no_grad():
                q_values = self.model(state)
                action = torch.argmax(q_values).item()
        return action

    def save_replay(self, currentState, action, reward, new_state, done):
        self.replayBuffer.append([currentState, action, reward, new_state, done])

    def update_weights(self, tau=None):
        if tau is None:
            tau = self.tau
        
        for target_param, local_param in zip(self.targetModel.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def train(self):
        if len(self.replayBuffer) < self.batch_size:
            return 0

        batch = random.sample(self.replayBuffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)

        # Get Q values for current states
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))

        # Get max Q values for next states from target model
        next_q_values = self.targetModel(next_states).max(1)[0].unsqueeze(1)

        # Compute target Q values
        target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))

        # Compute loss and update weights
        loss = self.criterion(current_q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()
        
        return loss.item()

    def save(self, name):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, f'./{name}.pth')

    def load(self, name):
        checkpoint = torch.load(f'./{name}.pth', weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', 1.0)
        self.update_weights()

def train_dqn(episodes=20000, steps=1000):

    env = gym.make("MountainCar-v0")
    agent = DQN(env=env)

    rewards_per_episode = np.zeros(episodes)
    success_count = 0
    training_losses = []
    epsilon_values = []

    start_time = time.time()

    for ep in tqdm(range(episodes)):
        state, _ = env.reset()
        total_reward = 0
        done = False

        for step in range(steps):
            action = agent.predict_action(state)
            new_state, reward, done, _, _ = env.step(action)

            # Reward shaping
            position, velocity = new_state
            custom_reward = abs(position - (-0.5)) * 2 # encourage moving towards goal
            if position >= 0.5:
                custom_reward += 10
            custom_reward -= 0.1

            # Save experience
            agent.save_replay(state, action, custom_reward, new_state, done)

            # Train agent
            loss = agent.train()
            if loss:
                training_losses.append(loss)

            total_reward += custom_reward
            state = new_state

            if done:
                break

        rewards_per_episode[ep] = total_reward

        if state[0] >= 0.5:
            success_count += 1


    success_rate = success_count / episodes
    end_time = time.time()
    training_time = end_time - start_time

    plt.plot(rewards_per_episode, label="Reward per Episode")
    plt.axhline(0, color='black', linestyle='--', label="Goal Reached")
    plt.title(f"Training Progress - Success Rate: {success_rate*100:.2f}%")
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"training_rewards.png")
    plt.show()

    agent.save(f"mountain_car_{ep}")
    print(f"Saved model at episode {ep}.")

    stats = {
        "model": agent,
        "training_time": round(training_time, 2),
    }

    return stats

def evaluate_dqn(agent, episodes=100, steps=1000):
    env = gym.make("MountainCar-v0")

    success_count = 0
    rewards_per_episode = []
    time_to_goal_per_episode = []

    agent.epsilon = 0  # stop exploration

    for ep in tqdm(range(episodes)):
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps_to_goal = 0

        for step in range(steps):
            action = agent.predict_action(state, evaluate=True)
            state, reward, done, _, _ = env.step(action)

            # Apply custom reward shaping
            position, velocity = state
            custom_reward = abs(position - (-0.5)) * 2 # encourage moving towards goal
            if position >= 0.5:
                custom_reward += 10 # reward for reaching the goal
            custom_reward -= 0.1 # small penalty for each step

            total_reward += custom_reward
            steps_to_goal += 1
            if done:
                break

        rewards_per_episode.append(total_reward)

        if state[0] >= 0.5:
            success_count += 1
            time_to_goal_per_episode.append(steps_to_goal)
        else:
            time_to_goal_per_episode.append(steps_to_goal)

    success_rate = success_count / episodes
    mean_rewards = np.mean(rewards_per_episode)
    avg_time_to_goal = np.mean(time_to_goal_per_episode) if time_to_goal_per_episode else 0

    print(f"Evaluation complete. Solved {success_count}/{episodes} episodes.")

    env.close()

    stats = {
        "success_rate": round(success_rate, 2),
        "rewards_per_episode": [round(r, 2) for r in rewards_per_episode],
        "mean_rewards": round(mean_rewards, 2),
        "average_time": round(avg_time_to_goal, 2),
    }

    return stats


if __name__ == "__main__":
    stats = train_dqn(episodes=1000)
    print(f"Training Stats: {stats}")

    eval_stats = evaluate_dqn(stats["model"])
    print(f"Evaluation Stats: {eval_stats}")
