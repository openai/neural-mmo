import numpy as np
from forge.blade.systems import ai
from forge.blade.lib.enums import Material, Neon
from forge.blade.action import action
from pdb import set_trace as T

class Stat:
   def __init__(self, val, maxVal):
      self._val = val
      self._max = maxVal

   def increment(self, amt=1):
      self._val = min(self.max, self.val + amt)

   def decrement(self, amt=1):
      self._val = max(0, self.val - amt)

   @property
   def val(self):
      return self._val 

   @property
   def max(self):
      return self._max

   def packet(self):
      return {'val': self.val, 'max': self.max}

class Player:
   public = set(
         'pos lastPos R C food water health entID annID name colorInd color timeAlive kill attackMap damage freeze immune'.split())

   def __init__(self, entID, color, config):
      self._config = config

      self._R, self._C = config.R, config.C
      self._pos = config.SPAWN()
      self._lastPos = self.pos

      self._food   = Stat(config.FOOD, config.FOOD)
      self._water  = Stat(config.WATER, config.WATER)
      self._health = Stat(config.HEALTH, config.HEALTH)

      self._entID = entID
      self._name = 'Neural_' + str(self._entID)
      self._timeAlive = 0

      self._damage = None
      self._freeze = 0
      self._immune = True
      self._kill = False

      self._annID, self._color = color
      self._colorInd = self._annID
      self._attackMap = np.zeros((7, 7, 3)).tolist()

      self._index = 1
      self._immuneTicks = 15

      self._actions = None
      self._attack  = None

   def __getattribute__(self, name):
      if name in Player.public:
         return getattr(self, '_' + name)
      return super().__getattribute__(name)

   def __setattr__(self, name, value):
      if name in Player.public:
         raise AttributeError('Property \"' + name + '\" is read only: agents cannot modify their server-side data')
      return super().__setattr__(name, value)

   def packet(self):
      data = {}
      for key in Player.public:
         val = getattr(self, key)
         data[key] = val 
         if hasattr(val, 'packet'):
            data[key] = val.packet()
      if self._attack is not None:  
         data['attack'] = {
               'style': self._attack.action.__name__,
               'target': self._attack.args.entID}
      return data

   #PCs interact with the world only through stimuli
   #to prevent cheating 
   def decide(self, packets):
      action, args = self.cpu.decide(self, packets)
      return action, args

   def forage(self, world):
      r, c = self._pos
      isForest = type(world.env.tiles[r, c].mat) in [Material.FOREST.value]
      if isForest and world.env.harvest(r, c):
         self.food.increment(5)

      isWater = Material.WATER.value in ai.adjacentMats(world.env, self._pos)
      if isWater:
         self.water.increment(5)

   def lavaKill(self, world):
      r, c = self._pos
      if type(world.env.tiles[r, c].mat) == Material.LAVA.value:
         self._kill = True
      return self._kill

   def updateStats(self):
      if (self._food.val > self._food.max//2 and
            self._water.val > self._water.max//2):
            self._health.increment()

      self._water.decrement()
      self._food.decrement()

      if self._food.val  <= 0:
         self._health.decrement()
      if self._water.val <= 0:
         self._health.decrement()

   def updateCounts(self, world):
      r, c = self._pos
      world.env.tiles[r, c].counts[self._colorInd] += 1

   def mapAttack(self):
      if self._attack is not None:
         attack = self._attack
         name = attack.action.__name__
         if name == 'Melee':
            attackInd = 0
         elif name == 'Range':
            attackInd = 1
         elif name == 'Mage':
            attackInd = 2
         rt, ct = attack.args.pos
         rs, cs = self._pos
         dr = rt - rs
         dc = ct - cs
         if abs(dr)<=3 and abs(dc)<=3:
            self._attackMap[3+dr][3+dc][attackInd] += 1

   def step(self, world):
      if not self.alive: return
      self._freeze = max(0, self._freeze-1)
      self.updateCounts(world)

      if self.lavaKill(world): return
      self.forage(world)
      self.updateStats()

      self._damage = None
      self._timeAlive += 1
      self.updateImmune()

   def act(self, world, actions, val):
      if not self.alive: return

      self._actions = actions
      self._attack  = actions[action.Attack]
      self.mapAttack()
      
      self.val = val
      self._lastPos = self._pos
      for meta, atnArgs in actions.items():
         atn, args = atnArgs.action, atnArgs.args
         if args is None:
            atn.call(world, self)
         else:
            atn.call(world, self, args)
         #atn.call(world, self, *args)

   @property
   def alive(self):
      return self._health.val > 0

   def updateImmune(self):
      if self._timeAlive >= self._immuneTicks:
         self._immune = False

   #Note: does not stack damage, but still applies to health
   def applyDamage(self, damage):
      if self.immune:
         return
      self._damage = damage
      self._health.decrement(damage)
