from pdb import set_trace as T
import numpy as np
import h5py

import nmmo


class OfflineDataset:
    def __init__(self, name):
        self.name = name

    def load(self):
        f = h5py.File(f'{self.name}.hdf5', 'r')
        self.obs = f.get('obs')
        self.atn = f.get('atn')
        self.rewards = f.get('rewards')
        self.dones = f.get('dones')
        return self

    def create(self, obs_dim, atn_dim, horizon, num_episodes, num_players):
        f = h5py.File(f'{self.name}.hdf5', 'w')
        self.obs = f.create_dataset('obs', (horizon, num_episodes, num_players, obs_dim), dtype='f')
        self.atn = f.create_dataset('atn', (horizon, num_episodes, num_players, atn_dim), dtype='i')
        self.rewards = f.create_dataset('rewards', (horizon, num_episodes, num_players), dtype='f')
        self.dones = f.create_dataset('dones', (horizon, num_episodes, num_players), dtype=np.dtype('bool'))
        return self

    def write(self, obs, atn, rewards, dones, t, episode):
        '''Write a single timestep of a single episode

        obs, rewards, dones are as returned from the env
        atn is the action dict submitted to the env
        t, episode are indices '''
        self.obs[t, episode] = np.stack(list(obs.values()))
        self.atn[t, episode] = np.stack(list(atn.values()))
        self.rewards[t, episode] = np.stack(list(rewards.values()))
        self.dones[t, episode] = np.stack(list(dones.values()))

    def write_vectorized(self, obs, atn, rewards, dones, t, episode_list):
        self.obs[t, episode_list] = obs
        self.atn[t, episode_list] = atn
        self.rewards[t, episode_list] = rewards
        self.dones[t, episode_list] = dones


EPISODES = 4
HORIZON = 16

config = nmmo.config.Default()
env = nmmo.integrations.CleanRLEnv(config)

print('Creating h5 dataset')
dataset = OfflineDataset('nmmo').create(
    obs_dim=env.observation_space(0).shape[0],
    atn_dim = env.action_space(0).shape[0],
    horizon=HORIZON,
    num_episodes=EPISODES,
    num_players=config.PLAYER_N
)

print('Collecting data')
for episode in range(EPISODES):
    print(f'Reset {episode}')
    obs = env.reset()
    for t in range(HORIZON):
        # Compute actions from network
        atn = {i+1: [0, 0, 0, 0, 0, 0, 0, 0] for i in range(config.PLAYER_N)}

        # Be sure to .copy() the atn dict -- nmmo modifies it in place
        nxt_obs, rewards, dones, infos = env.step(atn.copy())

        dataset.write(obs, atn, rewards, dones, t, episode)

        # Currently discards the last obs -- not sure if this is needed
        obs = nxt_obs

print('Loading h5 dataset')
dataset = OfflineDataset('nmmo').load()
print(dataset.obs)
print(dataset.atn)
print(dataset.rewards)
print(dataset.dones)
