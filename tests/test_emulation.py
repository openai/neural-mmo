from pdb import set_trace as T
import numpy as np

import nmmo

def init_env(config_cls=nmmo.config.Small):
    env = nmmo.Env(config_cls())                                                  
    obs = env.reset()                                                         
    return env, obs

def test_emulate_flat_obs():
    class Config(nmmo.config.Small):
        EMULATE_FLAT_OBS = True
   
    init_env(Config)

def test_emulate_flat_atn():
    class Config(nmmo.config.Small):
        EMULATE_FLAT_ATN = True
   
    init_env(Config)

def test_emulate_const_nent():
    class Config(nmmo.config.Small):
        EMULATE_CONST_NENT = True
   
    init_env(Config)

def test_all_emulation():
    class Config(nmmo.config.Small):
        EMULATE_FLAT_OBS  = True
        EMULATE_FLAT_ATN  = True
        EMULATE_CONST_POP = True
   
    init_env(Config)

def test_emulate_single_agent():
    class Config(nmmo.config.Small):
        EMULATE_CONST_NENT = True

    config = Config()
    envs   = nmmo.emulation.multiagent_to_singleagent(config)  

    for e in envs:
        ob = e.reset()
        for i in range(32):
            ob, reward, done, info = e.step({})

def equals(batch1, batch2):
   entity_keys = [e[0][0] for e in nmmo.io.stimulus.Serialized]
   assert list(batch1.keys()) == list(batch2.keys()) == entity_keys
   for (entity_name,), entity in nmmo.io.stimulus.Serialized:
        batch1_attrs = batch1[entity_name]
        batch2_attrs = batch2[entity_name]

        attr_keys = 'Continuous Discrete N'.split()
        assert list(batch1_attrs.keys()) == list(batch2_attrs.keys()) == attr_keys
                                                                              
        for key in attr_keys:
            assert np.array_equal(batch1_attrs[key], batch2_attrs[key])

def test_pack_unpack_obs():
    env, obs = init_env()
    packed   = nmmo.emulation.pack_obs(obs)
    packed   = np.vstack(list(packed.values()))
    T()
    unpacked = nmmo.emulation.unpack_obs(env.config, packed)
    batched  = nmmo.emulation.batch_obs(obs)

    equals(unpacked, batched)

def test_obs_pack_speed(benchmark):
    env, obs = init_env()
    benchmark(lambda: nmmo.emulation.pack_obs(obs))

def test_obs_unpack_speed(benchmark):
    env, obs = init_env()
    packed   = nmmo.emulation.pack_obs(obs)
    packed   = np.vstack(list(packed.values()))

    benchmark(lambda: nmmo.emulation.unpack_obs(env.config, packed))

if __name__ == '__main__':
   test_flat_obs()
