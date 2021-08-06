from pdb import set_trace as T

import math, random
import numpy as np
import itertools
import trueskill

from neural_mmo.forge.blade.lib import enums

class Agent:
    scripted = False
    name     = 'Neural_'
    color    = enums.Neon.CYAN
    pop      = 0
    rating   = trueskill.Rating()

    def __init__(self, config, idx):
       self.config = config
       self.iden   = idx
       self.pop    = Agent.pop


