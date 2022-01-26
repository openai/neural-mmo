from pdb import set_trace as T
import numpy as np

from enum import Enum, auto

import nmmo
from nmmo.lib import utils
from nmmo.lib.utils import staticproperty

class NodeType(Enum):
   #Tree edges
   STATIC = auto()    #Traverses all edges without decisions
   SELECTION = auto() #Picks an edge to follow

   #Executable actions
   ACTION    = auto() #No arguments
   CONSTANT  = auto() #Constant argument
   VARIABLE  = auto() #Variable argument

class Node(metaclass=utils.IterableNameComparable):
   @staticproperty
   def edges():
      return []

   #Fill these in
   @staticproperty
   def priority():
      return None

   @staticproperty
   def type():
      return None

   @staticproperty
   def leaf():
      return False

   @classmethod
   def N(cls, config):
      return len(cls.edges)

   def args(stim, entity, config):
      return []

class Fixed:
   pass

#ActionRoot
class Action(Node):
   nodeType = NodeType.SELECTION

   @staticproperty
   def edges():
      '''List of valid actions'''
      return [Move, Attack]

   @staticproperty
   def n():
      return len(Action.arguments)

   def args(stim, entity, config):
      return nmmo.Serialized.edges 

   #Called upon module import (see bottom of file)
   #Sets up serialization domain
   def hook():
      idx = 0
      arguments = []
      for action in Action.edges:
         for args in action.edges:
            if not 'edges' in args.__dict__:
               continue
            for arg in args.edges: 
               arguments.append(arg)
               arg.serial = tuple([idx])
               arg.idx = idx 
               idx += 1
      Action.arguments = arguments

class Move(Node):
   priority = 1
   nodeType = NodeType.SELECTION
   def call(env, entity, direction):
      r, c  = entity.pos
      entID = entity.entID
      entity.history.lastPos = (r, c)
      rDelta, cDelta = direction.delta
      rNew, cNew = r+rDelta, c+cDelta
      
      #One agent per cell
      tile = env.map.tiles[rNew, cNew] 
      if tile.occupied and not tile.lava:
         return

      if entity.status.freeze > 0:
         return

      env.dataframe.move(nmmo.Serialized.Entity, entID, (r, c), (rNew, cNew))
      entity.base.r.update(rNew)
      entity.base.c.update(cNew)

      env.map.tiles[r, c].delEnt(entID)
      env.map.tiles[rNew, cNew].addEnt(entity)

      if env.map.tiles[rNew, cNew].lava:
         entity.receiveDamage(None, entity.resources.health.val)

   @staticproperty
   def edges():
      return [Direction]

   @staticproperty
   def leaf():
      return True

class Direction(Node):
   argType = Fixed

   @staticproperty
   def edges():
      return [North, South, East, West]

   def args(stim, entity, config):
      return Direction.edges

class North(Node):
   delta = (-1, 0)

class South(Node):
   delta = (1, 0)

class East(Node):
   delta = (0, 1)

class West(Node):
   delta = (0, -1)


class Attack(Node):
   priority = 0
   nodeType = NodeType.SELECTION
   @staticproperty
   def n():
      return 3

   @staticproperty
   def edges():
      return [Style, Target]

   @staticproperty
   def leaf():
      return True

   def inRange(entity, stim, config, N):
      R, C = stim.shape
      R, C = R//2, C//2

      rets = set([entity])
      for r in range(R-N, R+N+1):
         for c in range(C-N, C+N+1):
            for e in stim[r, c].ents.values():
               rets.add(e)

      rets = list(rets)
      return rets

   def l1(pos, cent):
      r, c = pos
      rCent, cCent = cent
      return abs(r - rCent) + abs(c - cCent)

   def call(env, entity, style, targ):
      if entity.isPlayer and not env.config.game_system_enabled('Combat'):
         return 

      #Check if self targeted
      if entity.entID == targ.entID:
         return

      #ADDED: POPULATION IMMUNITY
      #if entity.population == targ.population:
      #   return

      #Check attack range
      rng     = style.attackRange(env.config)
      start   = np.array(entity.base.pos)
      end     = np.array(targ.base.pos)
      dif     = np.max(np.abs(start - end))

      #Can't attack same cell or out of range
      if dif == 0 or dif > rng:
         return 
      
      #Execute attack
      entity.history.attack = {}
      entity.history.attack['target'] = targ.entID
      entity.history.attack['style'] = style.__name__
      targ.attacker = entity
      targ.attackerID.update(entity.entID)

      from nmmo.systems import combat
      dmg = combat.attack(entity, targ, style.skill)

      if style.freeze and dmg > 0:
         targ.status.freeze.update(env.config.COMBAT_FREEZE_TIME)

      return dmg

class Style(Node):
   argType = Fixed
   @staticproperty
   def edges():
      return [Melee, Range, Mage]

   def args(stim, entity, config):
      return Style.edges


class Target(Node):
   argType = None
   #argType = Player 

   @classmethod
   def N(cls, config):
      #return config.WINDOW ** 2
      return config.N_AGENT_OBS

   def args(stim, entity, config):
      #Should pass max range?
      return Attack.inRange(entity, stim, config, None)

class Melee(Node):
   nodeType = NodeType.ACTION
   index = 0
   freeze=False

   def attackRange(config):
      return config.COMBAT_MELEE_REACH

   def skill(entity):
      return entity.skills.melee

class Range(Node):
   nodeType = NodeType.ACTION
   index = 1
   freeze=False

   def attackRange(config):
      return config.COMBAT_RANGE_REACH

   def skill(entity):
      return entity.skills.range

class Mage(Node):
   nodeType = NodeType.ACTION
   index = 2
   freeze=True

   def attackRange(config):
      return config.COMBAT_MAGE_REACH

   def skill(entity):
      return entity.skills.mage

class Inventory(Node):
    priority = 2
    argType  = Fixed

    @staticproperty
    def edges():
        return [InventoryAction, Item]

    def call(env, entity, inventory_action, item):
        if __debug__:
            assert inventory_action in (Use, Discard)

        if item not in entity.inventory:
            return

        if inventory_action == Use:
            return item.use(entity)

        return entity.inventory.remove(item)

class InventoryAction(Node):
    argType = Fixed

    @staticproperty
    def edges():
        return [Use, Discard]

    def args(stim, entity, config):
        return InventoryAction.edges

class Use(Node): pass
class Discard(Node): pass

class Item(Node):
    argType  = 'Entity'

    @classmethod
    def N(cls, config):
        return config.N_ITEM_OBS

    def args(stim, entity, config):
        return stim.exchange.items()

    @classmethod
    def gameObjects(cls, realm, entity, val):
        return [realm.entity(targ) for targ in entity.targets]

class Exchange(Node):
    priority = 3
    argType  = Fixed

    @staticproperty
    def edges():
        return [ExchangeAction, Item]#, Quantity, Price]

    def call(env, entity, exchange_action, item, quantity, price=None):
        #Do not process exchange actions on death tick
        if not entity.alive:
            return

        if __debug__:
            assert exchange_action in (Buy, Sell)

        if exchange_action == Buy:
            if not entity.inventory.space:
                return
            return env.exchange.buy(env, entity, item, quantity)

        return env.exchange.sell(env, entity, item, quantity, price)

    class ExchangeAction(Node):
        argType = Fixed

        @staticproperty
        def edges():
            return [Buy, Sell]

        def args(stim, entity, config):
            return ExchangeAction.edges

class Buy(Node): pass
class Sell(Node): pass

class Level(Node):
    argType = Fixed

    @staticproperty
    def edges():
        return [1, 5, 10, 15, 20]

    def args(stim, entity, config):
        return Level.edges

class Quantity(Node):
    argType = Fixed

    @staticproperty
    def edges():
        return [1, 5, 10, 50]

    def args(stim, entity, config):
        return Quantity.edges

class Price(Node):
    argType = Fixed

    @staticproperty
    def edges():
        return [1, 5, 10, 25, 50, 100]

    def args(stim, entity, config):
        return Price.edges

#TODO: Add communication
class Message:
   pass

#TODO: Solve AGI
class BecomeSkynet:
   pass

Action.hook()
