#Defines various handlers for correctly spawning/updating NPCs

from sim.handler.Handler import Handler
import numpy as np
from pdb import set_trace as T

class MaterialHandler(Handler):
   def __init__(self, env, entityType, material, maxSpawn=None):
      self.material= material
      super().__init__(env, entityType, maxSpawn=maxSpawn)

   def initSpawn(self, env):
      R, C = env.shape
      self.spawns = []
      for r in range(R):
         for c in range(C):
            if type(env[r, c].mat) == self.material.value:
               self.spawns += [(r, c)]
      
   def spawnPos(self):
      ind = np.random.randint(0, len(self.spawns))
      return self.spawns[ind]

   def stepAction(self, world, entity, ind):
      ret = self.entities[ind].act(world)

      if not self.entities[ind].isAlive():
         self.entities[ind].yieldDrops()
      self.removeIfDead(world, ind)
      return ret
