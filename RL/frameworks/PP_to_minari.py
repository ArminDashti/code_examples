import minari
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
env_id = 'LunarLander-v2'
dataset_id = 'LunarLander-v2_dataset'
#%%
vec_env = make_vec_env(env_id, n_envs=1)
model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=50000)
model.save(f'C:/Users/armin/desktop/{env_id}/') # Create a zip file called MountainCarContinuous

model = PPO.load(f"C:/Users/armin/desktop/{env_id}/") # Load a zip file called MountainCarContinuous
#%%
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
#%%
model = PPO.load(f"C:/Users/armin/desktop/{env_id}/") # Load a zip file called MountainCarContinuous
env = minari.DataCollector(gym.make(env_id))
env.reset()

for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rew, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        env.reset()

dataset = env.create_dataset(dataset_id=dataset_id)
