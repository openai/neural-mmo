from pdb import set_trace as T
import numpy as np

from collections import defaultdict
import itertools

import gym

import nmmo
from nmmo.infrastructure import DataType

def pad_const_pop(config, dummy_ob, obs, rewards, dones, infos):
    for i in range(1, config.NENT+1):                               
        dones[i] = False #No partial agent episodes                       
        if i not in obs:                                                  
            obs[i] = dummy_ob                                         
            rewards[i] = 0                                                 
            infos[i] = {}

def pack_atn_space(config):
   actions = defaultdict(dict)                                             
   for atn in sorted(nmmo.Action.edges):                                   
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
   for entity, obs in observation.items():                                 
      for attr_name, attr_box in obs.items():                              
         n += np.prod(observation[entity][attr_name].shape)                
                                                                           
   return gym.spaces.Box(low=-2**20, high=2**20, shape=(int(n),), dtype=DataType.CONTINUOUS)


def batch_obs(obs):
    batched = {}
    for (entity_name,), entity in nmmo.io.stimulus.Serialized:
        batched[entity_name] = {}
        for dtype in 'Continuous Discrete N'.split():
            attr_obs = [obs[k][entity_name][dtype] for k in obs]
            batched[entity_name][dtype] = np.stack(attr_obs, 0)

    return batched

def pack_obs(obs):
    packed = {}
    for key in obs:
        ary = []
        for ent_name, ent_attrs in obs[key].items():
            for attr_name, attr in ent_attrs.items():
                ary.append(attr.ravel())
        packed[key] = np.concatenate(ary) 

    return packed 

def unpack_obs(config, packed_obs):
    obs, idx = {}, 0
    batch = len(packed_obs)
    for (entity_name,), entity in nmmo.io.stimulus.Serialized:
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
