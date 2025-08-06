import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

class QNetwork(nn.Module):
    def __init__(self, input_dim=2, output_dim=4, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class DQNAgent:
    def __init__(self, env, buffer_size=10000, batch_size=64, gamma=0.99, lr=1e-3):
        self.env = env
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma

        self.model = QNetwork().to("cpu")
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def act(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.env.action_space)
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.model(state)
            return torch.argmax(q_values, dim=-1).item()

    def store(self, transition):
        self.buffer.append(transition)

    def sample_batch(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample_batch()

        q_values = self.model(states)
        next_q_values = self.model(next_states)

        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze()
        max_next_q_value = next_q_values.max(1)[0]
        target = rewards + self.gamma * max_next_q_value * (1 - dones)

        loss = F.mse_loss(q_value, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, num_episodes=500, max_steps=100):
        all_rewards = []
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            for step in range(max_steps):
                action = self.act(state)
                next_state, reward, done = self.env.step(action)
                self.store((state, action, reward, next_state, float(done)))

                state = next_state
                total_reward += reward

                loss = self.train_step()
                if done:
                    break

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            all_rewards.append(total_reward)

            print(f"Episode {episode+1} | Reward: {total_reward:.2f} | Epsilon: {self.epsilon:.3f} | Loss: {loss if loss else 0:.4f}")

        return all_rewards


from agent import SimpleEnv  # or however you defined it

if __name__ == "__main__":
    env = SimpleEnv(max_steps=100, render=True)
    agent = DQNAgent(env, buffer_size=10000, batch_size=4, gamma=0.8, lr=1e-3)

    rewards = agent.train(num_episodes=300)

    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Simple DQN Agent")
    plt.show()