'''Manual test for client connectivity'''

from pdb import set_trace as T
import pytest

from neural_mmo.infra.env import Env

if __name__ == '__main__':
    env = Env()
    env.config.RENDER = True

    env.reset()
    while True:
       env.render()
       env.step({})
