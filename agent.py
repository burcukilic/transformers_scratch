import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformer_decoder import DecisionTransformerDecoder
import random
from collections import deque
import numpy as np
class SimpleEnv():
    def __init__(self, max_steps=100):
        pygame.init()
        self.width, self.height = 400, 300
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.running = True
        self.agent_size = 20
        self.max_steps = max_steps
        self.action_space = np.array([1, 2, 3, 4])
        self.reset()

    def reset(self):
        self.agent_pos = [20, 20]  # fixed start
        self.goal_pos = [360, 260]  # fixed goal
        self.steps_taken = 0
        self.done = False
        return [self.agent_pos[0]/400, self.agent_pos[1]/300] 

    def step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        if self.done:
            return self.agent_pos, 0, True

        # Move agent
        step_size = 20
        if action == 1:  # up
            self.agent_pos[1] -= step_size
        elif action == 2:  # down
            self.agent_pos[1] += step_size
        elif action == 3:  # left
            self.agent_pos[0] -= step_size
        elif action == 4:  # right
            self.agent_pos[0] += step_size

        # Clamp to window
        self.agent_pos[0] = max(0, min(self.agent_pos[0], self.width - self.agent_size))
        self.agent_pos[1] = max(0, min(self.agent_pos[1], self.height - self.agent_size))

        # Check goal
        if self.agent_pos == self.goal_pos:
            distance_to_goal = 0
            reward = 1
            self.done = True
        else:
            # reward is negative for distance to goal
            distance_to_goal = ((self.agent_pos[0] - self.goal_pos[0]) ** 2 + (self.agent_pos[1] - self.goal_pos[1]) ** 2) ** 0.5
            reward = (500-distance_to_goal) / 500  # Normalize reward to [0, 1]

        self.steps_taken += 1
        if self.steps_taken >= self.max_steps:
            self.done = True

        # Draw
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (0, 255, 0), (*self.goal_pos, self.agent_size, self.agent_size))
        pygame.draw.rect(self.screen, (255, 0, 0), (*self.agent_pos, self.agent_size, self.agent_size))
        pygame.display.flip()
        self.clock.tick(10000)

        return [self.agent_pos[0]/400, self.agent_pos[1]/300], reward, self.done

    def close(self):
        pygame.quit()

class ReplayBuffer:
    def __init__(self, capacity=10000, max_seq_len=100):
        self.capacity = capacity
        self.max_seq_len = max_seq_len
        self.buffer = deque(maxlen=capacity)

    def add_episode(self, states, actions, rewards):
        episode = {
            "states": np.array(states[-self.max_seq_len:]),
            "actions": np.array(actions[-self.max_seq_len:]),
            "rewards": np.array(rewards[-self.max_seq_len:])
        }
        self.buffer.append(episode)

    def sample(self, batch_size=32):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        return batch

    def __len__(self):
        return len(self.buffer)

class DecisionTransformerAgent:
    def __init__(self, env, model, replay_buffer, device="cpu", max_seq_len=100):
        self.env = env
        self.model = model
        self.replay_buffer = replay_buffer
        self.device = device
        self.max_seq_len = max_seq_len

    def run_episode(self, epsilon=0.1, max_steps=100):
        self.model.eval()
        state = self.env.reset()
        done = False

        states, actions, rewards = [], [], []

        for t in range(max_steps):
            #state_norm = np.array(state, dtype=np.float32) / np.array([self.env.observation_space.high])  # normalize if needed
            state_norm = np.array(state, dtype=np.float32)  # No normalization in this simple case
            states.append(state_norm)

            # Prepare inputs
            s_seq = torch.tensor(states, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, T, 2)
            r_seq = torch.tensor(rewards, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, T)
            a_seq = torch.tensor(actions, dtype=torch.long).unsqueeze(0).to(self.device) if actions else torch.zeros((1, 0), dtype=torch.long).to(self.device)

            # Pad to fixed length
            pad_len = self.max_seq_len - s_seq.size(1)
            if pad_len > 0:
                s_seq = F.pad(s_seq, (0, 0, 0, pad_len))
                r_seq = F.pad(r_seq, (0, pad_len))
                a_seq = F.pad(a_seq, (0, pad_len))

            # Decide next action
            if len(actions) < 1 or random.random() < epsilon:
                action = np.random.choice(self.env.action_space)
            else:
                with torch.no_grad():
                    
                    # Pad action, reward, and state to predict next step
                    a_seq_input = F.pad(a_seq, (0, 1), value=0)
                    r_seq_input = F.pad(r_seq, (0, 1), value=65.0)
                    s_seq_input = F.pad(s_seq, (0, 0, 0, 1), value=0)  # Use last state for padding

                    s_seq_input = s_seq_input[:, :a_seq_input.shape[1], :]
                    r_seq_input = r_seq_input[:, :a_seq_input.shape[1]]

                    logits = self.model(a_seq_input, r_seq_input, s_seq_input)
                    probs = logits[0, a_seq.size(1)]  # Use prediction for next action
                    action = torch.argmax(probs).item()

            # Environment step
            next_state, reward, done = self.env.step(action)

            actions.append(action)
            rewards.append(reward)
            state = next_state

            if done:
                break

        self.replay_buffer.add_episode(states, actions, rewards)

    def train_step(self, batch_size=16):
        self.model.train()
        if len(self.replay_buffer) < batch_size:
            return  # not enough data yet

        batch = self.replay_buffer.sample(batch_size)
        
        losses = []
        for episode in batch:
            actions = torch.tensor(episode["actions"], dtype=torch.long).unsqueeze(0)  # (1, T)
            rewards = torch.tensor(episode["rewards"], dtype=torch.float32).unsqueeze(0)  # (1, T)
            states = torch.tensor(episode["states"], dtype=torch.float32).unsqueeze(0)  # (1, T, 2)

            # Inputs for predicting next action
            a_in = F.pad(actions[:, :-1], (0, 1), value=0)  # (1, T)
            r_in = F.pad(rewards[:, :-1], (0, 1), value=0.0)
            s_in = F.pad(states[:, :-1, :], (0, 0, 0, 1), value=0.0)

            # Targets: true next action
            target = actions  # Predict full sequence (shifted)

            loss = self.model.loss(a_in.to(self.device), r_in.to(self.device), s_in.to(self.device), target.to(self.device))
            losses.append(loss)

        return torch.stack(losses).mean()

if __name__ == "__main__":
    env = SimpleEnv()
    obs = env.reset()

    model = DecisionTransformerDecoder(vocab_size=5, embed_dim=8, seq_len=100, num_blocks=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epsilon = 1.0          # start with full exploration
    epsilon_min = 0.05
    epsilon_decay = 0.99  # decay per episode

    replay_buffer = ReplayBuffer(capacity=10000, max_seq_len=100)
    agent = DecisionTransformerAgent(env, model, replay_buffer)

    num_episodes = 1000
    for episode in range(num_episodes):
        agent.run_episode(epsilon=epsilon)
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay)  # decay epsilon

        loss = agent.train_step(batch_size=16)
        if loss is not None:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        ep_reward = sum(replay_buffer.buffer[-1]['rewards'])
        print(f"Episode {episode + 1}/{num_episodes} completed | epsilon: {epsilon:.2f} | reward: {ep_reward:.2f} | loss: {loss.item() if loss else 'N/A'}")



    env.close()