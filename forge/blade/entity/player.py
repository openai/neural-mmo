import numpy as np
from pdb import set_trace as T

from forge.blade.systems import ai
from forge.blade.lib.enums import Material, Neon

from forge.blade.io import Stimulus, StimHook
from forge.blade.io.action import static as action

from forge.blade.systems.skill import Skills
from forge.blade.systems.inventory import Inventory

class Base(StimHook):
   def __init__(self, config, iden, pop, name, color):
      super().__init__(Stimulus.Entity.Base, config)

      self.name  = name + str(iden)
      self.color = color
 
      self.alive = True

      r, c = config.SPAWN()
      self.r.update(r)
      self.c.update(c)

      self.population.update(pop)

   def update(self, ent, world, actions):
      if ent.resources.health.val <= 0:
         self.alive = False
         return

      r, c = self.pos
      if type(world.env.tiles[r, c].mat) == Material.LAVA.value:
         self.alive = False
         return

      #Update counts
      r, c = self.pos
      world.env.tiles[r, c].counts[self.population.val] += 1

   @property
   def pos(self):
      return self.r.val, self.c.val

   def packet(self):
      data = self.outputs(self.config)

      data['name']     = self.name
      data['color']    = self.color.packet()

      return data

class History(StimHook):
   def __init__(self, config):
      super().__init__(Stimulus.Entity.History, config)
      self.actions = None
      self.attack  = None

      self.attackMap = np.zeros((7, 7, 3)).tolist()
      self.lastPos = None

   def update(self, ent, world, actions):
      self.damage.update(None)
      self.actions = actions
      key = action.Attack

      self.timeAlive.increment()

      #need to rekey this
      if key in actions:
         self.attack, self.targ = actions[key]
         #self.mapAttack()
 
   def mapAttack(self):
      if self.attack is not None:
         attack = self.attack
         targ = self.targ
         name = attack.__name__
         if name == 'Melee':
            attackInd = 0
         elif name == 'Range':
            attackInd = 1
         elif name == 'Mage':
            attackInd = 2
         rt, ct = targ.args.pos
         rs, cs = self.pos
         dr = rt - rs
         dc = ct - cs
         if abs(dr)<=3 and abs(dc)<=3:
            self.attackMap[3+dr][3+dc][attackInd] += 1

   def packet(self):
      data = self.outputs(self.config)

      if self.attack is not None:  
         data['attack'] = self.attack
         #   {
         #      'style': self._attack.action.__name__,
         #      'target': self._attack.args.name}

      return data
 
class Resources(StimHook):
   def __init__(self, config):
      super().__init__(Stimulus.Entity.Resources, config)

   def update(self, ent, world, actions):
      pass

def wilderness(config, pos):
   rCent = config.R//2
   cCent = config.C//2

   R = abs(pos[0] - rCent)
   C = abs(pos[1] - cCent)

   diff = max(R, C) - 7
   diff = max(diff // 3, -1)
   return diff

class Status(StimHook):
   def __init__(self, config):
      super().__init__(Stimulus.Entity.Status, config)

   def update(self, ent, world, actions):
      self.immune.decrement()
      self.freeze.decrement()
   
      lvl = wilderness(self.config, ent.base.pos)
      self.wilderness.update(lvl)

class Player():
   def __init__(self, config, iden, pop, name='', color=None):
      self.config = config

      #Identifiers
      self.entID = iden 
      self.annID = pop

      #Submodules
      self.base      = Base(config, iden, pop, name, color)
      
      self.resources = Resources(config)
      self.status    = Status(config)
      self.skills    = Skills(config)
      self.history   = History(config)
      #self.inventory = Inventory(config)
      #self.chat      = Chat(config)

      #What the hell is this?
      #self._index = 1

   #Note: does not stack damage, but still applies to health
   def applyDamage(self, dmg, style):
      self.resources.food.increment(amt=dmg)
      self.resources.water.increment(amt=dmg)

      self.skills.applyDamage(dmg, style)
      
   def receiveDamage(self, dmg):
      self.resources.health.decrement(dmg)
      self.resources.food.decrement(amt=dmg)
      self.resources.water.decrement(amt=dmg)

      self.history.damage.update(dmg)
      self.skills.receiveDamage(dmg)

   @property
   def serial(self):
      return self.annID, self.entID
 
   def packet(self):
      data = {}

      data['entID']    = self.entID
      data['annID']    = self.annID

      data['base']     = self.base.packet()
      data['resource'] = self.resources.packet()
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

      self.resources.update(self, world, actions)
      self.status.update(self, world, actions)
      self.skills.update(self, world, actions)
      self.history.update(self, world, actions)
      #self.inventory.update(world, actions)
      #self.update(world, actions)

   def act(self, world, atnArgs):
      #Right now we only support one arg. So *args for future update
      atn, args = atnArgs
      atn.call(world, self, *args)


