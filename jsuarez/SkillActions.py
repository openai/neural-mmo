#Defines actions which entities may use to interact with the world.
#Does not validate whether an action is allowed; this should be
#handled prior to the function call. These are mainly passed as
#function handles within AI logic before being called during act().

from pdb import set_trace as T
import numpy as np
from sim.lib import Combat, Enums, AI, Utils
from sim.lib.Enums import Material
from sim.modules import Skill as skill
from sim.action import ActionTree
from sim.action.ActionTree import ActionNode, ActionLeaf
from copy import deepcopy

class Pass(ActionLeaf):
   def edges(self, stim, entity):
      return ActionTree.EmptyArgs()

class Reproduce(ActionLeaf):
   def __call__(self, world, entity):
      #if entity.food < entity.maxFood//2 or entity.water < entity.maxWater//2:
      #      return
      #posList = AI.adjacentEmptyPos(world.env, entity.pos)
      #if len(posList) == 0:
      #   return
       
      #posInd = np.random.choice(np.arange(len(posList)))
      #pos = posList[posInd]
      '''
      offspring = ServerPC(
            pos=entity.pos,
            health = entity.health,
            water = entity.water)
      entity.health = entity.health//2.0
      offspring.health = entity.health
      entity.water = entity.water//2.0
      offspring.water = entity.water
      return offspring
      '''
      pass

   def edges(self, stim, entity):
      return Args()

class Attack(ActionNode):
   def edges(self, world, entity):
      return [MeleeV2, Ranged]

class Ranged(ActionLeaf):
   def __call__(self, world, entity, targ):
      damage = Combat.attack(entity, targ, entity.skills.ranged)
      return damage

   def edges(self, world, entity):
      args = SetArgs()
      for e in AI.l1Range(world.env.ent, world.size,
            entity.pos, entity.hitRange):
         args.add(e)
      return args

class Skill(ActionNode):
   def edges(self, world, entity):
      return [Harvest, Process]

class Harvest(ActionNode):
   def edges(self, world, entity):
      return [Fish, Mine]

class Fish(ActionLeaf):
   def __call__(self, world, entity):
      entity.skills.fishing.harvest(entity.inv)

   def edges(self, world, entity):
      for mat in AI.adjacentMat(world.env.tiles, entity.pos):
         if mat == Material.WATER.value:
            return EmptyArgs()
 
class Mine(ActionLeaf):
   def __call__(self, world, entity, tile):
      world.env.harvest(tile.r, tile.c)
      entity.skills.mining.harvest(entity.inv)

   def edges(self, world, entity):
      r, c = entity.pos
      args = SetArgs()
      for tile in AI.adjacencyTiles(world.env.tiles, r, c):
         if type(tile.mat) == Material.OREROCK.value:
            args.add(tile)
      return args
      '''
      for mat in AI.adjacentMat(world.env.tiles, entity.pos):
         if mat == Material.OREROCK.value:
            return EmptyArgs()
      '''

class Process(ActionNode):
   def edges(self, world, entity):
      return [Cook, Smith]

class Cook(ActionLeaf):
   def __call__(self, world, entity, item):
      entity.skills.cooking.process(entity.inv)

   def edges(self, world, entity):
      args = DiscreteArgs()
      for itm in entity.inv:
         if itm.useSkill == skill.Cooking:
            args.add(Args([itm]))
      return args

class Smith(ActionLeaf):
   def __call__(self, world, entity, item):
      entity.skills.smithing.process(entity.inv)

   def edges(self, world, entity):
      args = DiscreteArgs()
      for itm in entity.inv:
         if itm.useSkill == skill.Smithing:
            args.add(enums.args([itm]))
      return args

'''
def buy(entity, world, item, quant, itemPrice):
   world.market.buy(item, quant, itemPrice)

def sell(entity, world, item, quant, itemPrice):
   world.market.sell(item, quant, itemPrice)

def cancelOffer(entity, world, offer):
   offer.cancel()

def message(entity, world, other, data):
   other.receiveMessage(entity, data)

#There are a huge number of possible exchanges.
#For effeciency, these are not validated a priori
#and will return False if invalid
def allowableExchanges(self, world, entity):
   ret = defaultdict(set)
   for itm in ItemHook.ItemList.items:
      for atn in (Actions.buy, Actions.sell):
         ret[atn].add(Enums.Args([itm, IntegerArg(), IntegerArg()]))
         ret[atn].add(Enums.Args([itm, IntegerArg(), IntegerArg()]))
   return ret
'''
   
