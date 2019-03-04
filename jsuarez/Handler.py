#Abstract class. Defines shared routines for managing and updating
#sets of entities within the world. Mainly subclassed by PC/NPC

from pdb import set_trace as T

class Handler():
   def __init__(self, env, entityType, maxSpawn=None):
      self.entities = []
      self.initSpawn(env)
      self.entityType = entityType
      self.maxSpawn = maxSpawn

   def addEntity(self, entity, ent):
      self.entities += [entity]
      r, c = entity.pos
      ent[r, c].append(entity)

   def step(self, world):
      for i in range(len(self.entities) - 1, -1, -1):
         self.stepAction(world, self.entities[i], i)
         self.stepEnv(world.env, self.entities[i])

   def stepEnv(self, env, entity):
      pass

   #override
   def stepAction(self, world, entity, ind):
      pass

   def removeIfDead(self, world, ind):
      if not self.entities[ind].isAlive(): 
         self.entities[ind].cpu.death()
         r, c = self.entities[ind].pos
         world.env.ent[r, c].remove(self.entities[ind])
         del self.entities[ind]
 
   def initSpawn(self, env):
      pass

   def spawnPos(self):
      pass
   
   def spawn(self, ent):
      if self.maxSpawn is None or len(self.entities) < self.maxSpawn:
         pos = self.spawnPos()
         entity = self.entityType(pos)
         self.addEntity(entity, ent) 

