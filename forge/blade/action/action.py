from pdb import set_trace as T
from forge.blade.action import tree
from forge.blade.lib import utils, enums
import numpy as np
from enum import Enum, auto

class staticproperty(property):
    def __get__(self, cls, owner):
        return self.fget.__get__(None, owner)()

class ActionArgs:
   def __init__(self, action, args):
      self.action = action
      self.args = args

class NodeType(Enum):
   #Tree edges
   STATIC = auto()    #Traverses all edges without decisions 
   SELECTION = auto() #Picks an edge to follow

   #Executable actions
   ACTION    = auto() #No arguments
   CONSTANT  = auto() #Constant argument
   VARIABLE  = auto() #Variable argument

class Node:
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
   def args(stim, entity, config):
      return None

class ActionRoot(Node):
   nodeType = NodeType.STATIC
   @staticproperty
   def edges():
      return [Move, Attack]
      #return [Move, Attack, Exchange, Skill]

class Move(Node):
   priority = 1
   nodeType = NodeType.SELECTION
   def call(world, entity, rDelta, cDelta):
      r, c = entity.pos
      rNew, cNew = r+rDelta, c+cDelta
      if world.env.tiles[rNew, cNew].state.index in enums.IMPASSIBLE:
         return
      if not utils.inBounds(rNew, cNew, world.shape):
         return
      if entity.freeze > 0:
         return

      entity._pos = rNew, cNew
      entID = entity.entID
      
      r, c = entity.lastPos
      world.env.tiles[r, c].delEnt(entID)

      r, c = entity.pos
      world.env.tiles[r, c].addEnt(entID, entity)

   @staticproperty
   def edges():
      return [Pass, North, South, East, West]

   def args(stim, entity, config):
      return Move.edges

class Pass(Node):
   nodeType = NodeType.ACTION
   def call(world, entity):
      Move.call(world, entity, 0, 0)

class North(Node):
   nodeType = NodeType.ACTION
   def call(world, entity):
      Move.call(world, entity, -1, 0)

class South(Node):
   nodeType = NodeType.ACTION
   def call(world, entity):
      Move.call(world, entity, 1, 0)

class East(Node):
   nodeType = NodeType.ACTION
   def call(world, entity):
      Move.call(world, entity, 0, 1)

class West(Node):
   nodeType = NodeType.ACTION
   def call(world, entity):
      Move.call(world, entity, 0, -1)

class Attack(Node):
   nodeType = NodeType.SELECTION
   @staticproperty
   def n():
      return 3

   @staticproperty
   def edges():
      return [Melee, Range, Mage]

   def inRange(entity, stim, N):
      R, C = stim.shape
      R, C = R//2, C//2
      #R, C = entity.pos

      rets = []
      for r in range(R-N, R+N+1):
         for c in range(C-N, C+N+1):
            for e in stim[r, c].ents.values():
               rets.append(e)
      return rets

   def l1(pos, cent):
      r, c = pos
      rCent, cCent = cent
      return abs(r - rCent) + abs(c - cCent)

   def call(world, entity, targ, damageF, freeze=False):
      if entity.entID == targ.entID:
         entity._attack = None
         return
      #entity.targPos = targ.pos
      #entity.attkPos = entity.lastPos
      #entity.targ = targ
      damage = damageF(entity, targ)
      assert type(damage) == int
      if freeze and damage > 0:
         targ._freeze = 3
      return
      #return damage

   def args(stim, entity, config):
      return [Melee, Range, Mage]
      #return Melee.args(stim, entity, config) + Range.args(stim, entity, config) + Mage.args(stim, entity, config)

class AttackStyle(Node):
   pass

class Melee(Node):
   nodeType = NodeType.VARIABLE
   index = 0
   @staticproperty
   def edges():
      return None

   priority = 2
   def call(world, entity, targ):
      damageF = world.config.MELEEDAMAGE
      Attack.call(world, entity, targ, damageF)

   def args(stim, entity, config):
      return Attack.inRange(entity, stim, config.MELEERANGE)

class Range(Node):
   nodeType = NodeType.VARIABLE
   index = 1
   @staticproperty
   def edges():
      return None

   priority = 2
   def call(world, entity, targ):
      damageF = world.config.RANGEDAMAGE
      Attack.call(world, entity, targ, damageF)

   def args(stim, entity, config):
      return Attack.inRange(entity, stim, config.RANGERANGE)

class Mage(Node):
   nodeType = NodeType.VARIABLE
   index = 2
   @staticproperty
   def edges():
      return None

   priority = 2
   def call(world, entity, targ):
      damageF = world.config.MAGEDAMAGE
      dmg = Attack.call(world, entity, targ, damageF, freeze=True)

   def args(stim, entity, config):
      return Attack.inRange(entity, stim, config.MAGERANGE)

class Reproduce:
   pass

class Skill(Node):
   nodeType = NodeType.SELECTION
   @staticproperty
   def edges():
      return [Harvest, Process]

   def args(stim, entity, config):
      return Skill.edges

class Harvest(Node):
   nodeType = NodeType.SELECTION
   @staticproperty
   def edges():
      return [Fish, Mine]

   def args(stim, entity, config):
      return Harvest.edges

class Fish(Node):
   nodeType = NodeType.ACTION

class Mine(Node):
   nodeType = NodeType.ACTION

class Process(Node):
   nodeType = NodeType.SELECTION
   @staticproperty
   def edges():
      return [Cook, Smith]

   def args(stim, entity, config):
      return Process.edges

class Cook(Node):
   nodeType = NodeType.ACTION

class Smith(Node):
   nodeType = NodeType.ACTION

class Exchange(Node):
   nodeType = NodeType.SELECTION
   @staticproperty
   def edges():
      return [Buy, Sell, CancelOffer, Pass]

   def args(stim, entity, config):
      return Exchange.edges

class Buy(Node):
   nodeType = NodeType.ACTION

class Sell(Node):
   nodeType = NodeType.ACTION

class CancelOffer(Node):
   nodeType = NodeType.ACTION

class Message:
   pass
