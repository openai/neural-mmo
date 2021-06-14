class Observation:
   def __init__(self, config, obs):
      self.config = config
      self.obs    = obs
      self.delta  = config.NSTIM
      self.tiles  = self.obs['Tile']['Continuous']
      self.agents = self.obs['Entity']['Continuous']
      self.n      = int(self.obs['Entity']['N'])

   def tile(self, rDelta, cDelta):
      return self.tiles[self.config.WINDOW * (self.delta + rDelta) + self.delta + cDelta]

   @property
   def agent(self):
      return self.agents[0]

   @property
   def agentID(self):
      pass

   @staticmethod
   def attribute(ary, attr):
      return float(ary[attr.index])
