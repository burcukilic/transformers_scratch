import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformer_decoder import DecisionTransformerDecoder
import random

class SimpleEnv():
    def __init__(self, max_steps=100):
        pygame.init()
        self.width, self.height = 400, 300
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.running = True
        self.agent_size = 20
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.agent_pos = [20, 20]  # fixed start
        self.goal_pos = [360, 260]  # fixed goal
        self.steps_taken = 0
        self.done = False
        return self.agent_pos

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
        self.clock.tick(1000)

        return self.agent_pos, reward, self.done

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    env = SimpleEnv()
    obs = env.reset()

    model = DecisionTransformerDecoder(vocab_size=5, embed_dim=16, seq_len=100, num_blocks=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epsilon = 1.0          # start with full exploration
    epsilon_min = 0.05
    epsilon_decay = 0.998  # decay per episode

    rewards, actions = [], []

    while env.running:
        if env.done or len(actions) >= 100:
            # Learning step after episode ends
            if len(actions) > 1:
                seq_len = 100
                # Prepare input and target
                input_actions = torch.tensor(actions[:-1])
                target_actions = torch.tensor(actions[1:])
                reward_seq = torch.tensor(rewards[:-1])

                # Pad
                input_actions = F.pad(input_actions, (0, seq_len - input_actions.size(0)), value=0)
                target_actions = F.pad(target_actions, (0, seq_len - target_actions.size(0)), value=0)
                reward_seq = F.pad(reward_seq, (0, seq_len - reward_seq.size(0)), value=0.0)

                # Train step
                input_actions = input_actions.unsqueeze(0).long()       # (1, seq_len)
                target_actions = target_actions.unsqueeze(0).long()     # (1, seq_len)
                reward_seq = reward_seq.unsqueeze(0).float()            # (1, seq_len)

                output = model(input_actions, reward_seq)
                loss = model.loss(input_actions, reward_seq, target_actions)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                print(f"Episode done | Loss: {loss.item():.4f} | Epsilon: {epsilon:.3f} | Rewards: {sum(rewards):.2f}")

            # Reset
            rewards, actions = [], []
            obs = env.reset()
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # ================== Decide Next Action ==================
        if len(actions) < 1 or random.random() < epsilon:
            action = random.choice([1, 2, 3, 4])  # explore
        else:
            # Use model prediction
            with torch.no_grad():
                r_seq = torch.tensor(rewards).unsqueeze(0).float()
                # add 1 more reward to r_seq of '1'
                r_seq = torch.cat([r_seq, torch.tensor([[1.0]])], dim=1)
                a_seq = torch.tensor(actions).unsqueeze(0).long()
                # Pad
                r_seq = F.pad(r_seq, (0, 100 - r_seq.size(1)))
                a_seq = F.pad(a_seq, (0, 100 - a_seq.size(1)))
                logits = model(a_seq, r_seq)  # (1, seq, vocab)
                next_logits = logits[0, len(actions)]  # next timestep
                action = torch.argmax(next_logits).item()  # exploit

        # ================ Step the environment ==================
        obs, reward, done = env.step(action)
        rewards.append(reward)
        actions.append(action)

    env.close()