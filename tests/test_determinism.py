from pdb import set_trace as T

import nmmo


def test_determinism():
    env1 = nmmo.Env()
    env1.reset(seed=42)
    for i in range(32):
        env1.step({})

    env2 = nmmo.Env()
    env2.reset(seed=42)
    for i in range(32):
        env2.step({})

    npc1 = env1.realm.npcs.values()
    npc2 = env2.realm.npcs.values()

    for n1, n2 in zip(npc1, npc2):
        assert n1.pos == n2.pos
