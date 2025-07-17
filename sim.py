
import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

env = gym.make('AdroitHandDoor-v1',  render_mode='human')

# render and step through the environment
env.reset()
action_dim = env.action_space.shape[0]
observation_dim = env.observation_space.shape[0]


for _ in range(1000):
    env.render()
    action = env.action_space.sample()  # Sample a random action
    obs, reward, terminated, truncated, info = env.step(action)
    print(reward)
    if terminated or truncated:
        env.reset()

env.close()

