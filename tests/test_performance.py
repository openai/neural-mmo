from pdb import set_trace as T
import pytest

from neural_mmo.infra.env import Env
from neural_mmo.core.config import Config, SmallMaps, Resource, Combat, Progression, NPC, AllGameSystems

# Test utils
def create_and_reset(conf):
    env = Env(conf())
    env.reset(idx=1)

def create_config(base, *systems):
    systems   = (base, *systems)
    name      = '_'.join(cls.__name__ for cls in systems)

    conf                    = type(name, systems, {})()

    conf.TERRAIN_TRAIN_MAPS = 1
    conf.TERRAIN_EVAL_MAPS  = 1

    return conf

def benchmark_config(benchmark, base, nent, *systems):
    conf      = create_config(base, *systems)
    conf.NENT = nent

    env = Env(conf)
    env.reset()

    benchmark(env.step, actions={})

def benchmark_env(benchmark, env, nent):
    env.config.NENT = nent
    env.reset()

    benchmark(env.step, actions={})

# Small map tests -- fast with greater coverage for individual game systems
def test_small_env_creation(benchmark):
    benchmark(lambda: Env(SmallMaps()))

def test_small_env_reset(benchmark):
    env = Env(SmallMaps())
    benchmark(lambda: env.reset(idx=1))

def test_fps_small_base_1_pop(benchmark):
    benchmark_config(benchmark, SmallMaps, 1) 

def test_fps_small_resource_1_pop(benchmark):
    benchmark_config(benchmark, SmallMaps, 1, Resource) 

def test_fps_small_combat_1_pop(benchmark):
    benchmark_config(benchmark, SmallMaps, 1, Combat) 

def test_fps_small_progression_1_pop(benchmark):
    benchmark_config(benchmark, SmallMaps, 1, Progression) 

def test_fps_small_rcp_1_pop(benchmark):
    benchmark_config(benchmark, SmallMaps, 1, Resource, Combat, Progression) 

def test_fps_small_npc_1_pop(benchmark):
    benchmark_config(benchmark, SmallMaps, 1, NPC)

def test_fps_small_all_1_pop(benchmark):
    benchmark_config(benchmark, SmallMaps, 1, AllGameSystems)

def test_fps_small_rcp_100_pop(benchmark):
    benchmark_config(benchmark, SmallMaps, 100, Resource, Combat, Progression) 

def test_fps_small_all_100_pop(benchmark):
    benchmark_config(benchmark, SmallMaps, 100, AllGameSystems)

# Reuse large maps since we aren't benchmarking the reset function
def test_large_env_creation(benchmark):
    benchmark(lambda: Env(Config()))

def test_large_env_reset(benchmark):
    env = Env(Config())
    benchmark(lambda: env.reset(idx=1))

LargeMapsRCP = Env(create_config(Config, Resource, Combat, Progression))
LargeMapsAll = Env(create_config(Config, AllGameSystems))

def test_fps_large_rcp_1_pop(benchmark):
    benchmark_env(benchmark, LargeMapsRCP, 1)

def test_fps_large_rcp_100_pop(benchmark):
    benchmark_env(benchmark, LargeMapsRCP, 100)

def test_fps_large_rcp_1000_pop(benchmark):
    benchmark_env(benchmark, LargeMapsRCP, 1000)

def test_fps_large_all_1_pop(benchmark):
    benchmark_env(benchmark, LargeMapsAll, 1)

def test_fps_large_all_100_pop(benchmark):
    benchmark_env(benchmark, LargeMapsAll, 100)

def test_fps_large_all_1000_pop(benchmark):
    benchmark_env(benchmark, LargeMapsAll, 1000)
