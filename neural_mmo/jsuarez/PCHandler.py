#Handler for PCs. Unlike NPCs, the NPC handler must manage environment
#updates, including food/water levels. Also handles reproduction.

from pdb import set_trace as T
from collections import defaultdict
from sim.handler.Handler import Handler
from sim.lib import Enums, AI, Actions
from sim.entity.PC import PC
from sim.modules import Skill, ItemHook
from sim.lib.Enums import Material

class PCHandler(Handler):
   def __init__(self, env, cpu):
      super().__init__(env, cpu.ann)
      self.cpu = cpu
      
   #Const for now. Better later
   def initSpawn(self, env):
      self.spawnCent = 13

   def spawnPos(self):
      return self.spawnCent, self.spawnCent

   def stepEnv(self, env, entity):
      r, c = entity.pos
      tile = env.tiles[r, c]
      '''
      if tile.mat.harvestable:
         for drop in env.harvest(r, c):
            entity.inv.add(*drop)
      '''
      #if (type(env.tiles[r, c].mat) in [Material.FOREST.value]):
      if (entity.food < entity.maxFood and type(env.tiles[r, c].mat) in [Material.FOREST.value]):
         if env.harvest(r, c):
            entity.food += 1
      if (entity.water < entity.maxWater and 
            Material.WATER.value in 
            AI.adjacentMats(env, entity.pos)):
         entity.water += 1

   def stepAction(self, world, entity, ind):
      actions = Actions.ActionTree(world, entity)
      ret = self.entities[ind].act(world, actions)

      self.removeIfDead(world, ind)
      '''
      if ret not in (-1, None):
         pos = AI.randAdjacent(*self.entities[ind].pos)
         entity = PC(ret, pos)
         self.addEntity(entity, world.ent)
      '''

   def spawn(self, ent):
      pos = self.spawnPos()
      entity = self.cpu.spawn()
      entityWrapper = PC(entity, self.spawnPos())
      self.addEntity(entityWrapper, ent)


