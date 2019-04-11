from forge.ethyr.torch import utils as tu
from forge.ethyr import stim

from pdb import set_trace as T

class Stim:
   def __init__(self, ent, env, config):
      sz = config.STIM
      conv, ents = stim.environment(env, ent, sz, config)
      flat = stim.entity(ent, ent, config)

      self.flat = tu.var(flat)
      self.conv = tu.var(conv)
      self.ents = tu.var(ents)
