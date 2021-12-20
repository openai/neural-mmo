'''Manual test for client connectivity'''

from pdb import set_trace as T
import pytest

import nmmo

if __name__ == '__main__':
    env = nmmo.Env()
    env.config.RENDER = True

    env.reset()
    while True:
       env.render()
       env.step({})
