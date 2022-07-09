from pdb import set_trace as T
import numpy as np

from collections import defaultdict
import itertools

import gym

import nmmo
from nmmo.infrastructure import DataType

class SingleAgentEnv:
    def __init__(self, env, idx, max_idx):
        self.config = env.config
        self.env    = env
        self.idx    = idx
        self.last   = idx == max_idx 

    def reset(self):
        if not self.env.has_reset:
            self.obs = self.env.reset()

        return self.obs[self.idx]            

    def step(self, actions):
        if self.last:
            self.obs, self.rewards, self.dones, self.infos = self.env.step(actions)

        i = self.idx
        return self.obs[i], self.rewards[i], self.dones[i], self.infos[i]

def multiagent_to_singleagent(config):
    assert config.EMULATE_CONST_PLAYER_N, "Wrapper requires constant num agents"

    base_env = nmmo.Env(config)
    n = config.PLAYER_N

    return [SingleAgentEnv(base_env, i, n) for i in range(1, n+1)]
        
def pad_const_nent(config, dummy_ob, obs, rewards, dones, infos):
    for i in range(1, config.PLAYER_N+1):                               
        if i not in obs:                                                  
            obs[i]     = dummy_ob                                         
            rewards[i] = 0                                                 
            infos[i]   = {}
            dones[i]   = False

def const_horizon(dones):
    for agent in dones:
        dones[agent] = True

    return dones

def pack_atn_space(config):
   actions = defaultdict(dict)                                             
   for atn in sorted(nmmo.Action.edges(config)):
      for arg in sorted(atn.edges):
         actions[atn][arg] = arg.N(config)

   n = 0                                                                   
   flat_actions = {}                                                  
   for atn, args in actions.items():                                       
       ranges = [range(e) for e in args.values()]                          
       for vals in itertools.product(*ranges):                                       
          flat_actions[n] = {atn: {arg: val for arg, val in zip(args, vals)}}
          n += 1 

   return flat_actions

def pack_obs_space(observation):
   n = 0                                                                   
   #for entity, obs in observation.items():                                 
   for entity in observation:                                 
      obs = observation[entity]
      #for attr_name, attr_box in obs.items():                              
      for attr_name in obs:                              
         attr_box = obs[attr_name]
         n += np.prod(observation[entity][attr_name].shape)                
                                                                           
   return gym.spaces.Box(
           low=-2**20, high=2**20,
           shape=(int(n),), dtype=DataType.CONTINUOUS)


def batch_obs(config, obs):
    batched = {}
    for (entity_name,), entity in nmmo.io.stimulus.Serialized:
        if not entity.enabled(config):
            continue

        batched[entity_name] = {}
        for dtype in 'Continuous Discrete N'.split():
            attr_obs = [obs[k][entity_name][dtype] for k in obs]
            batched[entity_name][dtype] = np.stack(attr_obs, 0)

    return batched

def pack_obs(obs):
    packed = {}
    for key in obs:
        ary = []
        obs[key].items()
        for ent_name, ent_attrs in obs[key].items():
            for attr_name, attr in ent_attrs.items():
                ary.append(attr.ravel())
        packed[key] = np.concatenate(ary) 

    return packed 

def unpack_obs(config, packed_obs):
    obs, idx = {}, 0
    batch = len(packed_obs)
    for (entity_name,), entity in nmmo.io.stimulus.Serialized:
        if not entity.enabled(config):
            continue

        n_entity = entity.N(config)
        n_continuous, n_discrete = 0, 0
        obs[entity_name] = {}

        for attribute_name, attribute in entity:
            if attribute.CONTINUOUS:
                n_continuous += 1
            if attribute.DISCRETE:
                n_discrete += 1

        inc = int(n_entity * n_continuous)
        obs[entity_name]['Continuous'] = packed_obs[:, idx: idx + inc].reshape(batch, n_entity, n_continuous)
        idx += inc

        inc = int(n_entity * n_discrete)
        obs[entity_name]['Discrete'] = packed_obs[:, idx: idx + inc].reshape(batch, n_entity, n_discrete)
        idx += inc

        inc = 1
        obs[entity_name]['N'] = packed_obs[:, idx: idx + inc].reshape(batch, 1)
        idx += inc

    return obs
