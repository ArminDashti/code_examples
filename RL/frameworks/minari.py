'''
pip install minari
https://minari.farama.org/
https://github.com/Farama-Foundation/Minari
'''

import minari
#%%
# Download dataset
dataset = minari.load_dataset('door-expert-v1', download=True).iterate_episodes()
#%%
# Dataset to Dict
dataset = minari.load_dataset('door-expert-v2', download=True)
episodes = []
for episode_data in self.dataset.iterate_episodes():
    episode = [] 
    for step in range(episode_data.total_timesteps):
        sample = {}
        sample['observation'] = episode_data.observations[step, :]
        sample['action'] = episode_data.actions[step]
        sample['reward'] = episode_data.rewards[step]
        if (step == episode_data.total_timesteps-1):
            sample['next_observation'] = episode_data.observations[step]
        else:
            sample['next_observation'] = episode_data.observations[step+1]
            sample['termination'] = episode_data.terminations[step]
            sample['truncation'] = episode_data.truncations[step]
            sample['info'] = episode_data.infos['success'][step]
            episode.append(sample)
            episodes.append(episode)
#%%