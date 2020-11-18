from pdb import set_trace as T
import numpy as np

from forge.blade.systems import skill, droptable, combat
from forge.blade.lib.enums import Material, Neon

from forge.blade.io.action import static as Action
from forge.blade.io.stimulus import Static

class Resource:
   def __init__(self, val, mmax=None):
      self._val = val
      if mmax is None:
         self._max = val
      else:
         self._max = mmax

   @property
   def max(self):
      return self._max

   @property
   def empty(self):
      return self._val == 0

   def packet(self):
      return { 'val': self._val, 'max': self._max}

   def increment(self, val):
      self._val = min(self._val + val, self._max)

   def decrement(self, val):
      self._val = max(self._val - val, 0)

   def __eq__(self, val):
      return self._val == val

   def __ne__(self, val):
      return self._val != val

   def __gt__(self, val):
      return self._val > val

   def __ge__(self, val):
      return self._val >= val

   def __lt__(self, val):
      return self._val < val

   def __le__(self, val):
      return self._val <= val

class Status:
   def __init__(self, ent):
      self.config = ent.config

      self.wilderness = Static.Entity.Wilderness(ent.dataframe, ent.entID)
      self.immune     = Static.Entity.Immune(    ent.dataframe, ent.entID)
      self.freeze     = Static.Entity.Freeze(    ent.dataframe, ent.entID)

   def update(self, realm, entity, actions):
      self.immune.decrement()
      self.freeze.decrement()

      wilderness = combat.wilderness(self.config, entity.base.pos)
      self.wilderness.update(wilderness)

   def packet(self):
      data = {}
      data['wilderness'] = self.wilderness.val
      data['immune']     = self.immune.val
      data['freeze']     = self.freeze.val
      return data

class History:
   def __init__(self, ent):
      self.actions = None
      self.attack  = None

      self.damage    = Static.Entity.Damage(   ent.dataframe, ent.entID)
      self.timeAlive = Static.Entity.TimeAlive(ent.dataframe, ent.entID)

      self.attackMap = np.zeros((7, 7, 3)).tolist()
      self.lastPos = None

   def update(self, realm, entity, actions):
      self.damage.update(0)
      self.actions = actions

      #No way around this circular import I can see :/
      from forge.blade.io.action import static as action
      key = action.Attack

      self.timeAlive.increment()

      '''
      if key in actions:
         action = actions[key]
      '''

   def packet(self):
      data = {}
      data['damage']    = self.damage.val
      data['timaAlive'] = self.timeAlive.val

      if self.attack is not None:
         data['attack'] = self.attack

      return data

class Base:
   def __init__(self, ent, pos, iden, name, color):
      self.color = color
      r, c       = pos

      self.r          = Static.Entity.R(ent.dataframe, ent.entID, r)
      self.c          = Static.Entity.C(ent.dataframe, ent.entID, c)

      self.population = Static.Entity.Population(ent.dataframe, ent.entID, 0)
      self.self       = Static.Entity.Self(      ent.dataframe, ent.entID, True)


   def update(self, realm, entity, actions):
      r, c = self.pos
      if type(realm.map.tiles[r, c].mat) == Material.LAVA.value:
         entity.killed = True
         return

   @property
   def pos(self):
      return self.r.val, self.c.val

   def packet(self):
      data = {}

      data['r']          = self.r.val
      data['c']          = self.c.val
      data['name']       = self.name
      data['color']      = self.color.packet()

      return data


class Entity:
   def __init__(self, realm, iden):
      self.dataframe    = realm.dataframe
      self.config       = realm.config
      self.entID        = iden

      self.repr         = None
      self.killed       = False
      self.vision       = 5

      self.attacker     = None
      self.target       = None
      self.closest      = None

   def packet(self):
      data = {}

      data['status']  = self.status.packet()
      data['history'] = self.history.packet()
      data['loadout'] = self.loadout.packet()
      data['alive']   = self.alive

      return data

   def step(self, realm, actions):
      self.base.update(realm, self, actions)
      self.status.update(realm, self, actions)
      self.history.update(realm, self, actions)

   def receiveDamage(self, source, dmg):
      pass

   def applyDamage(self, dmg, style):
      pass

   @property
   def pos(self):
      return self.base.pos

   @property
   def alive(self):
      if self.killed:
         return False
      return True
