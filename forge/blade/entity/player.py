import numpy as np
from pdb import set_trace as T

from forge.blade.systems import ai
from forge.blade.lib.enums import Material, Neon

from forge.blade.io import Stimulus, StimHook
from forge.blade.io.action import static as action

from forge.blade.systems.skill import Skills
from forge.blade.systems.inventory import Inventory

#Makes private attributes read only
class Protected:
   def __getattribute__(self, name):
      try:
         return super().__getattribute__('_' + name)
      except AttributeError:
         return super().__getattribute__(name)

   def __setattr__(self, name, value):
      if hasattr(self, '_' + name):
         raise AttributeError('Property \"' + name + '\" is read only')
      return super().__setattr__(name, value)

class Base(Protected, StimHook):
   def __init__(self, config, iden, pop, name, color):
      super().__init__(Stimulus.Entity.Base, config)

      self._name  = name + str(iden)
      self._color = color
 
      self._alive = True

      r, c = config.SPAWN()
      self.r.update(r)
      self.c.update(c)

      self.population.update(pop)

   def update(self, ent, world, actions):
      if ent.resource.health <= 0:
         self._alive = False
         return

      r, c = self.pos
      if type(world.env.tiles[r, c].mat) == Material.LAVA.value:
         self._alive = False
         return

      #Update counts
      r, c = self.pos
      world.env.tiles[r, c].counts[self._population.val] += 1

   @property
   def pos(self):
      return self._r.val, self._c.val

   def packet(self):
      data = self.outputs(self.config)

      data['name']     = self._name
      data['color']    = self._color.packet()

      return data

class History(Protected, StimHook):
   def __init__(self, config):
      super().__init__(Stimulus.Entity.History, config)
      self._actions = None
      self._attack  = None

      self._attackMap = np.zeros((7, 7, 3)).tolist()
      self._lastPos = None

   def update(self, ent, world, actions):
      self._damage.update(None)
      self._actions = actions
      key = action.Attack
      if key in actions:
         self._attack = actions[key]
         self.mapAttack()
 
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

   def packet(self):
      data = self.outputs(self.config)

      if self._attack is not None:  
         data['attack'] = {
               'style': self._attack.action.__name__,
               'target': self._attack.args.name}

      return data
 
class Resource(Protected, StimHook):
   def __init__(self, config):
      super().__init__(Stimulus.Entity.Resource, config)

   def update(self, ent, world, actions):
      if (self._food.val > self._food.max//2 and
            self._water.val > self._water.max//2):
            self._health.increment()

      self.forage(ent, world)
      ent.history.timeAlive.increment()

      self.water.decrement()
      self.food.decrement()

      if self.food.val  <= 0:
         self.health.decrement()
      if self.water.val <= 0:
         self.health.decrement()

   def forage(self, ent, world):
      r, c = ent.base.pos
      isForest = type(world.env.tiles[r, c].mat) in [Material.FOREST.value]
      if isForest and world.env.harvest(r, c):
         self.food.increment(5)

      isWater = Material.WATER.value in ai.adjacentMats(
         world.env, ent.base.pos)
      if isWater:
         self.water.increment(5)

   #Note: does not stack damage, but still applies to health
   def applyDamage(self, damage):
      if self.immune:
         return

      self._damage = damage
      self._health.decrement(damage)

class Status(Protected, StimHook):
   def __init__(self, config):
      super().__init__(Stimulus.Entity.Status, config)

   def update(self, ent, world, actions):
      self.immune.decrement()
      self.freeze.decrement()

class Player(Protected):
   def __init__(self, config, iden, pop, name='', color=None):
      self._config = config

      #Identifiers
      self._entID = iden 
      self._annID = pop

      #Submodules
      self.base      = Base(config, iden, pop, name, color)
      self.resource  = Resource(config)
      self.status    = Status(config)
      self.skills    = Skills(config)
      self.history   = History(config)
      #self.inventory = Inventory(config)
      #self.chat      = Chat(config)

      #What the hell is this?
      #self._index = 1

   @property
   def serial(self):
      return self.annID, self.entID
 
   def packet(self):
      data = {}

      data['entID']    = self.entID
      data['annID']    = self.annID

      data['base']     = self.base.packet()
      data['resource'] = self.resource.packet()
      data['status']   = self.status.packet()
      data['skills']   = self.skills.packet()
      data['history']  = self.history.packet()

      return data
  
   #PCs interact with the world only through stimuli
   #to prevent cheating 
   def decide(self, packets):
      action, args = self.cpu.decide(self, packets)
      return action, args

   def step(self, world, actions):
      self.base.update(self, world, actions)
      if not self.base.alive:
         return

      self.resource.update(self, world, actions)
      self.status.update(self, world, actions)
      self.skills.update(world, actions)
      self.history.update(self, world, actions)
      #self.inventory.update(world, actions)
      #self.update(world, actions)

   def act(self, world, atnArgs):
      #Right now we only support one arg. So *args for future update
      atn, args = atnArgs.action, atnArgs.args
      if args is None:
         atn.call(world, self)
      else:
         atn.call(world, self, args)
      #atn.call(world, self, *args)


