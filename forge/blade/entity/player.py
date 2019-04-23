import numpy as np
from pdb import set_trace as T

from forge.blade.systems import ai
from forge.blade.lib.enums import Material, Neon

from forge.blade.io import action

class Player:
   def __init__(self, entID, color, config):
      self._config = config
      self.inputs(config)

      r, c = config.SPAWN()
      self.r.update(r)
      self.c.update(c)

      self._lastPos = self.pos

      self._entID = entID
      self._name = 'Neural_' + str(self._entID)
      self._kill = False

      self._annID, self._color = color
      self._colorInd = self._annID
      self._attackMap = np.zeros((7, 7, 3)).tolist()

      self._index = 1
      self._actions = None
      self._attack  = None

   def __getattribute__(self, name):
      try:
         return super().__getattribute__('_' + name)
      except AttributeError:
         return super().__getattribute__(name)

   def __setattr__(self, name, value):
      if hasattr(self, '_' + name):
         raise AttributeError('Property \"' + name + '\" is read only')
      return super().__setattr__(name, value)

   def inputs(self, config):
      for name, cls in config.static.Entity:
         setattr(self, '_'+cls.name, cls(config))

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
   
   @property
   def pos(self):
      return self._r.val, self._c.val

   #PCs interact with the world only through stimuli
   #to prevent cheating 
   def decide(self, packets):
      action, args = self.cpu.decide(self, packets)
      return action, args

   def forage(self, world):
      r, c = self.pos
      isForest = type(world.env.tiles[r, c].mat) in [Material.FOREST.value]
      if isForest and world.env.harvest(r, c):
         self.food.increment(5)

      isWater = Material.WATER.value in ai.adjacentMats(world.env, self.pos)
      if isWater:
         self.water.increment(5)

   def lavaKill(self, world):
      r, c = self.pos
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
      r, c = self.pos
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
         rs, cs = self.pos
         dr = rt - rs
         dc = ct - cs
         if abs(dr)<=3 and abs(dc)<=3:
            self._attackMap[3+dr][3+dc][attackInd] += 1

   def step(self, world):
      if not self.alive: return
      self._freeze.decrement()
      self.updateCounts(world)

      if self.lavaKill(world): return
      self.forage(world)
      self.updateStats()

      self._damage.update(None)
      self._timeAlive += 1
      self.updateImmune()

   def act(self, world, actions, val):
      if not self.alive: return

      self._actions = actions
      self._attack  = actions[action.static.Attack]
      self.mapAttack()
      
      self.val = val
      self._lastPos = self.pos
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
      self.immune.decrement()

   #Note: does not stack damage, but still applies to health
   def applyDamage(self, damage):
      if self.immune:
         return
      self._damage = damage
      self._health.decrement(damage)
