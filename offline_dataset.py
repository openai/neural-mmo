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

    def write(self, t, episode, obs=None, atn=None, rewards=None, dones=None):
        '''Write a single timestep of a single episode

        obs, rewards, dones are as returned from the env
        atn is the action dict submitted to the env
        t, episode are indices '''
        if obs is not None:
            self.obs[t, episode] = np.stack(list(obs.values()))
        if atn is not None:
            self.atn[t, episode] = np.stack(list(atn.values()))
        if rewards is not None:
            self.rewards[t, episode] = np.stack(list(rewards.values()))
        if dones is not None:
            self.dones[t, episode] = np.stack(list(dones.values()))

    def write_vectorized(self, t, episode, obs=None, atn=None, rewards=None, dones=None):
        if obs is not None:
            self.obs[t, episode_list] = obs
        if atn is not None:
            self.atn[t, episode_list] = atn
        if rewards is not None:
            self.rewards[t, episode_list] = rewards
        if dones is not None:
            self.dones[t, episode_list] = dones

EPISODES = 5
HORIZON = 16
#EPISODES = 1000
#HORIZON = 1023

actions = np.load('actions.npy', allow_pickle=True)

config = nmmo.config.Default()
env = nmmo.integrations.CleanRLEnv(config, seed=42)

print('Creating h5 dataset')
dataset = OfflineDataset('nmmo0').create(
    obs_dim=env.observation_space(0).shape[0],
    atn_dim = env.action_space(0).shape[0],
    horizon=HORIZON,
    num_episodes=EPISODES,
    num_players=config.PLAYER_N
)

print('Collecting Actions')
for episode in range(EPISODES):
    print(f'Reset {episode}')
    obs = env.reset()
    for t in range(HORIZON):
        # Compute actions from network
        #atn = {i+1: [0, 0, 0, 0, 0, 0, 0, 0] for i in range(config.PLAYER_N)}
        dataset.write(t, episode, atn=actions[t])

        # Be sure to .copy() the atn dict -- nmmo modifies it in place
        obs, rewards, dones, infos = env.step({})

print('Generating Dataset')
for episode in range(EPISODES):
    print(f'Reset {episode}')
    obs = env.reset()
    for t in range(HORIZON):
        # Retrieve action from dataset
        atn = {i+1: e for i, e in enumerate(dataset.atn[t, episode])}

        # Be sure to .copy() the atn dict -- nmmo modifies it in place
        nxt_obs, rewards, dones, infos = env.step({})

        dataset.write(t, episode, obs=obs, rewards=rewards, dones=dones)

        # Currently discards the last obs -- not sure if this is needed
        obs = nxt_obs

env = nmmo.integrations.CleanRLEnv(config, seed=42)

print('Creating h5 dataset')
dataset = OfflineDataset('nmmo1').create(
    obs_dim=env.observation_space(0).shape[0],
    atn_dim = env.action_space(0).shape[0],
    horizon=HORIZON,
    num_episodes=EPISODES,
    num_players=config.PLAYER_N
)

print('Collecting Actions')
for episode in range(EPISODES):
    print(f'Reset {episode}')
    obs = env.reset()
    for t in range(HORIZON):
        # Compute actions from network
        atn = {i+1: [0, 0, 0, 0, 0, 0, 0, 0] for i in range(config.PLAYER_N)}
        dataset.write(t, episode, atn=actions[t])

        # Be sure to .copy() the atn dict -- nmmo modifies it in place
        obs, rewards, dones, infos = env.step({})

print('Generating Dataset')
for episode in range(EPISODES):
    print(f'Reset {episode}')
    obs = env.reset()
    for t in range(HORIZON):
        # Retrieve action from dataset
        atn = {i+1: e for i, e in enumerate(dataset.atn[t, episode])}

        # Be sure to .copy() the atn dict -- nmmo modifies it in place
        nxt_obs, rewards, dones, infos = env.step({})

        dataset.write(t, episode, obs=obs, rewards=rewards, dones=dones)

        # Currently discards the last obs -- not sure if this is needed
        obs = nxt_obs


print('Loading h5 dataset')
dataset = OfflineDataset('nmmo0').load()
print(dataset.obs)
print(dataset.atn)
print(dataset.rewards)
print(dataset.dones)

dataset1 = OfflineDataset('nmmo1').load()
print(np.sum(dataset.obs[:] != dataset1.obs[:]))
print(np.sum(dataset.atn[:] != dataset1.atn[:]))
print(np.sum(dataset.rewards[:] != dataset1.rewards[:]))
print(np.sum(dataset.dones[:] != dataset1.dones[:]))
