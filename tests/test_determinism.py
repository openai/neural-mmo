from pdb import set_trace as T
import numpy as np
import random

import nmmo


def test_determinism():
    config = nmmo.config.Default()
    env1 = nmmo.Env(config, seed=42)
    env1.reset()
    for i in range(2):
        obs1, _, _, _ = env1.step({})
    
    config = nmmo.config.Default()
    env2 = nmmo.Env(config, seed=42)
    env2.reset()
    for i in range(2):
        obs2, _, _, _ = env2.step({})

    npc1 = env1.realm.npcs.values()
    npc2 = env2.realm.npcs.values()

    for n1, n2 in zip(npc1, npc2):
        assert n1.pos == n2.pos

    assert list(obs1.keys()) == list(obs2.keys())
    keys = list(obs1.keys())
    for k in keys:
        ent1 = obs1[k]
        ent2 = obs2[k]
        
        obj = ent1.keys()
        for o in obj:
            obj1 = ent1[o]
            obj2 = ent2[o]

            attrs = list(obj1)
            for a in attrs:
                attr1 = obj1[a]
                attr2 = obj2[a]

                if np.sum(attr1 != attr2) > 0:
                    T()
                assert np.sum(attr1 != attr2) == 0

test_determinism()