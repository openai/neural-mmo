from pdb import set_trace as T
import numpy as np

import nmmo
from nmmo.io.stimulus import Serialized

def batch(obs):
    batched = {}
    for (entity_name,), entity in Serialized:
        batched[entity_name] = {}
        for dtype in 'Continuous Discrete N'.split():
            attr_obs = [obs[k][entity_name][dtype] for k in obs]
            batched[entity_name][dtype] = np.stack(attr_obs, 0)

    return batched


def pack(obs):
    packed = {}
    for key in obs:
        ary = []
        for ent_name, ent_attrs in obs[key].items():
            for attr_name, attr in ent_attrs.items():
                ary.append(attr.ravel())
        packed[key] = np.concatenate(ary) 

    return packed 

def unpack(config, packed_obs):
    obs, idx = {}, 0
    batch = len(packed_obs)
    for (entity_name,), entity in Serialized:
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

def equals(batch1, batch2):
   entity_keys = [e[0][0] for e in Serialized]
   assert list(batch1.keys()) == list(batch2.keys()) == entity_keys
   for (entity_name,), entity in Serialized:
        #attribute_keys = [e[0] for e in entity]

        batch1_attrs = batch1[entity_name]
        batch2_attrs = batch2[entity_name]

        attr_keys = 'Continuous Discrete N'.split()
        assert list(batch1_attrs.keys()) == list(batch2_attrs.keys()) == attr_keys 

        for key in attr_keys:
            assert np.array_equal(batch1_attrs[key], batch2_attrs[key])

class Config(nmmo.config.Medium):
    @property
    def SPAWN(self):
        return self.SPAWN_CONCURRENT

def setup():
    env = nmmo.Env(Config())
    obs = env.reset()
    return env, obs

def test_env_step_speed(benchmark):
    env, _ = setup()
    benchmark(lambda: env.step({}))

def test_obs_pack_speed(benchmark):
    _, obs = setup()
    benchmark(lambda: pack(obs))

def test_obs_unpack_speed(benchmark):
    env, obs = setup()
    packed = pack(obs)
    packed = np.vstack(list(packed.values()))
    benchmark(lambda: unpack(env.config, packed))

def test_pack_unpack():
    env, obs   = setup()
    packed   = pack(obs)
    packed   = np.vstack(list(packed.values()))
    unpacked = unpack(env.config, packed)
    batched  = batch(obs)

    equals(unpacked, batched)
