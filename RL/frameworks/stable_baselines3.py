'''
pip install stable-baselines3[extra]
https://stable-baselines3.readthedocs.io/
https://github.com/DLR-RM/stable-baselines3
'''

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
#%%
'''
https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
'''
vec_env = make_vec_env('LunarLander-v2', n_envs=1)
model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=50000)
model.save('C:/Users/armin/desktop/MountainCarContinuous/') # Create a zip file called MountainCarContinuous

model = PPO.load("C:/Users/armin/desktop/MountainCarContinuous/") # Load a zip file called MountainCarContinuous

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
#%%
