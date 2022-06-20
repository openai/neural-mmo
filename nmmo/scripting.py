from pdb import set_trace as T

class Observation:
   '''Unwraps observation tensors for use with scripted agents'''
   def __init__(self, config, obs):
      '''
      Args:
          config: A forge.blade.core.Config object or subclass object
          obs: An observation object from the environment
      '''
      self.config = config
      self.obs    = obs
      self.delta  = config.PLAYER_VISION_RADIUS
      self.tiles  = self.obs['Tile']['Continuous']

      n = int(self.obs['Entity']['N'])
      self.agents  = self.obs['Entity']['Continuous'][:n]
      self.n = n

      if config.ITEM_SYSTEM_ENABLED:
          n = int(self.obs['Item']['N'])
          self.items   = self.obs['Item']['Continuous'][:n]

      if config.EXCHANGE_SYSTEM_ENABLED:
          n = int(self.obs['Market']['N'])
          self.market = self.obs['Market']['Continuous'][:n]

   def tile(self, rDelta, cDelta):
      '''Return the array object corresponding to a nearby tile
      
      Args:
         rDelta: row offset from current agent
         cDelta: col offset from current agent

      Returns:
         Vector corresponding to the specified tile
      '''
      return self.tiles[self.config.PLAYER_VISION_DIAMETER * (self.delta + rDelta) + self.delta + cDelta]

   @property
   def agent(self):
      '''Return the array object corresponding to the current agent'''
      return self.agents[0]

   @staticmethod
   def attribute(ary, attr):
      '''Return an attribute of a game object

      Args:
         ary: The array corresponding to a game object
         attr: A forge.blade.io.stimulus.static stimulus class
      '''
      return float(ary[attr.index])
