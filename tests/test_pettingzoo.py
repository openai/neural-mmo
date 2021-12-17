from pdb import set_trace as T

from neural_mmo.infra.env import Env
from pettingzoo.test import parallel_api_test

env = Env()
parallel_api_test(env, num_cycles=2000)
