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
      self.delta  = config.NSTIM
      self.tiles  = self.obs['Tile']['Continuous']
      self.agents = self.obs['Entity']['Continuous']
      self.n      = int(self.obs['Entity']['N'])

   def tile(self, rDelta, cDelta):
      '''Return the array object corresponding to a nearby tile
      
      Args:
         rDelta: row offset from current agent
         cDelta: col offset from current agent

      Returns:
         Vector corresponding to the specified tile
      '''
      return self.tiles[self.config.WINDOW * (self.delta + rDelta) + self.delta + cDelta]

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
