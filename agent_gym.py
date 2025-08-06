import torch
import torch.nn.functional as F
import torch.optim as optim
from transformer_decoder import DecisionTransformer, DiscreteDecisionTransformer
import random
import numpy as np
from replay_buffer import ReplayBuffer
from matplotlib import pyplot as plt
import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)


class DecisionTransformerAgent:
    def __init__(self, env, model, replay_buffer, device="cpu", max_seq_len=100):
        self.env = env
        self.model = model
        self.replay_buffer = replay_buffer
        self.device = device
        self.max_seq_len = max_seq_len
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space['observation'].shape[0]# + env.observation_space['desired_goal'].shape[0]


    def run_episode(self, epsilon=0.1, max_steps=100, task=1):
        self.model.eval()
        goal_cell = [[1,3], [3, 1], [3, 2], [2, 3]][task]
        reset_cell = [3, 3]
        state, success = env.reset(options={'goal_cell': goal_cell, 'reset_cell': reset_cell})  # Set a desired goal
        #state = np.concatenate((state['observation'], state['desired_goal']), axis=-1)
        state = state['observation']

        terminated, truncated = False, False

        states, actions, rewards, tasks = [np.array(state)], [], [], []

        target_return = 1  # Fixed target return for simplicity
        current_target_return = target_return

        for _ in range(max_steps):

            # Decide next action
            if len(actions) < 1 or random.random() < epsilon:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    
                    # 1. Prepare states tensor
                    s_padded = np.zeros((self.max_seq_len, self.state_dim), dtype=np.float32)
                    s_hist = np.array(states)
                    s_len = min(len(s_hist), self.max_seq_len)
                    s_padded[-s_len:] = s_hist[-s_len:]
                    s_tensor = torch.from_numpy(s_padded).unsqueeze(0).to(self.device)

                    # 2. Prepare actions tensor
                    a_padded = np.zeros((self.max_seq_len, self.action_dim), dtype=np.float32)
                    a_hist = np.array(actions)
                    a_len = min(len(a_hist), self.max_seq_len)
                    a_padded[-a_len:] = a_hist[-a_len:]
                    a_tensor = torch.from_numpy(a_padded).unsqueeze(0).to(self.device)

                    # 3. Prepare task tensor
                    task_tensor = torch.full((1, self.max_seq_len), task, dtype=torch.long).to(self.device)

                    # 4. Prepare returns-to-go tensor
                    rtg_tensor = torch.full((1, self.max_seq_len, 1), current_target_return, dtype=torch.float32).to(self.device)
                    
                    # Get prediction from model
                    pred_action_seq = self.model(a_tensor, rtg_tensor, s_tensor, task_tensor)

                    # The prediction for the current step is the last one in the sequence
                    action = pred_action_seq[0, -1].cpu().numpy()
                    action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

            # Environment step
            obs, reward, terminated, truncated, _ = self.env.step(action)
            #obs = np.concatenate((obs['observation'], obs['desired_goal']), axis=-1)
            obs = obs['observation']

            actions.append(action.astype(np.float32))
            states.append(obs)
            rewards.append(reward)
            tasks.append(task)
            state = obs

            current_target_return -= reward

            if terminated or truncated:
                break

        rtg = self.compute_rtg(rewards)
        self.replay_buffer.add_episode(states[:-1], actions, rewards, rtg, tasks)

    def compute_rtg(self, rewards, gamma=0.99):
        rtg = []
        ret = 0
        for r in reversed(rewards):
            ret = r + gamma * ret
            rtg.insert(0, ret)
        return rtg

    def train_step(self, batch_size=16):
        self.model.train()
        if len(self.replay_buffer) < batch_size:
            return None  # Not enough data yet

        # CORRECTED: Vectorized batch processing
        batch = self.replay_buffer.sample(batch_size)
        
        # Placeholders for the batch
        s_batch, a_batch, rtg_batch, task_batch = [], [], [], []

        for episode in batch:
            ep_len = len(episode['states'])
            
            # Pad sequences for this episode to max_seq_len
            s_padded = np.zeros((self.max_seq_len, self.state_dim), dtype=np.float32)
            a_padded = np.zeros((self.max_seq_len, self.action_dim), dtype=np.float32)
            rtg_padded = np.zeros((self.max_seq_len, 1), dtype=np.float32)
            task_padded = np.zeros((self.max_seq_len,), dtype=np.long)

            # Right-align the data
            s_padded[-ep_len:] = episode['states']
            a_padded[-ep_len:] = episode['actions']
            rtg_padded[-ep_len:] = np.array(episode['returns_to_go']).reshape(-1, 1)
            task_padded[-ep_len:] = np.array(episode['tasks']).astype(np.int64)

            s_batch.append(s_padded)
            a_batch.append(a_padded)
            rtg_batch.append(rtg_padded)
            task_batch.append(task_padded)

        # Convert lists of arrays to single batch tensors
        s_tensor = torch.from_numpy(np.array(s_batch)).to(self.device)
        a_tensor = torch.from_numpy(np.array(a_batch)).to(self.device)
        rtg_tensor = torch.from_numpy(np.array(rtg_batch)).to(self.device)
        task_tensor = torch.from_numpy(np.array(task_batch)).to(self.device)
        
        # The target for prediction is the original action sequence
        target_a_tensor = a_tensor
        
        # The input action sequence is shifted by one timestep
        # input at t is a_{t-1}, with a_{-1} being a zero vector
        input_a_tensor = F.pad(a_tensor[:, :-1], (0, 0, 1, 0), value=0.0)

        # Forward pass
        # Assumes model.loss takes (input_actions, rtgs, states, target_actions)
        loss = self.model.loss(input_a_tensor, rtg_tensor, s_tensor, task_tensor, target_a_tensor)

        return loss

if __name__ == "__main__":

    max_steps = 60
    batch_size = 64
    lr = 1e-3

    example_map = [[1, 1, 1, 1, 1],
       [1, 0, 0, 'g', 1],
       [1, 0, 0, 0, 1],
       [1, 0, 0, 'r', 1],
       [1, 1, 1, 1, 1]
   ]

    env = gym.make('PointMaze_UMaze-v3', render_mode="human", maze_map=example_map, max_episode_steps=max_steps, reward_type='dense')

    obs, success = env.reset(options={'goal_cell': [1, 3], 'reset_cell': [3, 3]})  # Set a desired goal
    # concatenate observation and the desired goal
    #obs = np.concatenate((obs['observation'], obs['desired_goal']), axis=-1)
    obs = obs['observation']
    print(f"Observation shape: {obs.shape}, Action space shape: {env.action_space.shape}")

    model = DiscreteDecisionTransformer(action_dim=env.action_space.shape[0], state_dim=obs.shape[0], task_dim=2, embed_dim=8, seq_len=max_steps, num_blocks=5)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.9995

    replay_buffer = ReplayBuffer(capacity=10000, max_seq_len=max_steps)
    agent = DecisionTransformerAgent(env, model, replay_buffer, max_seq_len=max_steps)

    all_rewards = []
    num_episodes = 2000
    for episode in range(num_episodes):
        agent.run_episode(epsilon=epsilon, max_steps=max_steps, task=episode % 2)
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        loss = agent.train_step(batch_size=batch_size)
        if loss is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        ep_reward = sum(replay_buffer.buffer[-1]['rewards'])
        print(f"Episode {episode + 1}/{num_episodes} completed | epsilon: {epsilon:.2f} | reward: {ep_reward:.2f} | loss: {loss.item() if loss else 'N/A'}")

        all_rewards.append(ep_reward)

    env.close()

    # plot the rewards for each task separately
    plt.figure(figsize=(12, 6))
    for task in range(2):
        plt.plot([r for i, r in enumerate(all_rewards) if i % 2 == task], label=f'Task {task}')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards Over Episodes')
    plt.legend()
    plt.show()
