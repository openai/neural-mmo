from neural_mmo.forge.trinity.env import Env
from projekt.config import CompetitionRound1

from pettingzoo.test import parallel_api_test

config = CompetitionRound1()
env = Env(config)

parallel_api_test(env, num_cycles=2000)
