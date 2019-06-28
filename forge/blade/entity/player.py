import numpy as np
from pdb import set_trace as T

from forge.blade.systems import ai
from forge.blade.lib.enums import Material, Neon

from forge.blade.io import Stimulus
from forge.blade.io.action import static as action

class Player:
   def __init__(self, config, iden, pop, name='', color=None):
      self._config = config
      self.inputs(config)

      r, c = config.SPAWN()
      self.r.update(r)
      self.c.update(c)

      self._lastPos = self.pos

      self._entID = iden 
      self._name = name + str(iden)
      self._kill = False

      self._color = color
      self._annID = pop
      
      self._attackMap = np.zeros((7, 7, 3)).tolist()

      self._index = 1
      self._actions = None
      self._attack  = None

      self._population.update(self._annID)

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
      for name, cls in Stimulus.Entity:
         setattr(self, '_'+cls.name, cls(config))

   def outputs(self, config):
      data = {}
      for name, cls in Stimulus.Entity:
         data[name.lower()] = getattr(self, '_'+cls.name).packet()
      return data
 
   def packet(self):
      data = self.outputs(self.config)
      data['color'] = self._color.packet()
      data['name']  = self._name
      if self._attack is not None:  
         data['attack'] = {
               'style': self._attack.action.__name__,
               'target': self._attack.args.name}
      return data

   @property
   def serial(self):
      return self.annID, self.entID
   
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
      world.env.tiles[r, c].counts[self.population.val] += 1

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

   def step(self, world, actions):
      if not self.alive: return
      self._freeze.decrement()
      self.updateCounts(world)

      if self.lavaKill(world): return
      self.forage(world)
      self.updateStats()

      self._damage.update(None)
      self._timeAlive += 1
      self.updateImmune()

      self._actions = actions
      
      key = action.Attack
      if key in actions:
         self._attack = actions[key]
         self.mapAttack()
 
   def act(self, world, atnArgs):
      #Right now we only support one arg. So *args for future update
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
