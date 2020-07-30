import numpy as np
from pdb import set_trace as T

from forge.blade.systems import ai
from forge.blade.lib.enums import Material, Neon

from forge.blade.systems.skill import Skills
from forge.blade.systems.inventory import Inventory

class Base:
   def __init__(self, config, iden, pop, name, color):
      self.name  = name + str(iden)
      self.color = color
      self.self  = True
 
      self.alive = True

      self.r, self.c = config.SPAWN()
      self.population = pop

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
      world.env.tiles[r, c].counts[self.population] += 1

   @property
   def pos(self):
      return self.r, self.c

   def packet(self):
      #data = self.outputs(self.config)

      data = {}
      data['r']          = self.r
      data['c']          = self.c
      data['population'] = self.population
      data['self']       = self.self
      data['name']       = self.name
      data['color']      = self.color.packet()

      return data

class History:
   def __init__(self, config):
      self.timeAlive = 0
      self.actions = None
      self.attack  = None
      self.damage = None

      self.attackMap = np.zeros((7, 7, 3)).tolist()
      self.lastPos = None

   def update(self, ent, world, actions):
      self.damage = None
      self.actions = actions

      #No way around this circular import I can see :/
      from forge.blade.io.action import static as action
      key = action.Attack

      self.timeAlive += 1

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
      #data = self.outputs(self.config)
      data = {}
      data['damage']    = self.damage
      data['timaAlive'] = self.timeAlive

      if self.attack is not None:  
         data['attack'] = self.attack
         #   {
         #      'style': self._attack.action.__name__,
         #      'target': self._attack.args.name}

      return data

class Resource:
    def __init__(self, val):
        self.val = val
        self.max = val

    def packet(self):
        return {
                'val': self.val,
                'max': self.max}
 
#Todo: fix negative rollover
class Resources:
   def __init__(self, config):
      self.health = Resource(config.HEALTH)
      self.water  = Resource(config.RESOURCE)
      self.food   = Resource(config.RESOURCE)

   def update(self, ent, world, actions):
      self.health.max = ent.skills.constitution.level
      self.water.max  = ent.skills.fishing.level
      self.food.max   = ent.skills.hunting.level

   def packet(self):
      data = {}
      data['health'] = self.health.packet()
      data['food']   = self.food.packet()
      data['water']  = self.water.packet()
      return data

def wilderness(config, pos):
   rCent = config.R//2
   cCent = config.C//2

   R = abs(pos[0] - rCent)
   C = abs(pos[1] - cCent)

   #Circle crop with 0 starting at 10 squares from
   #center and increasing one level every 5 tiles
   wild = np.sqrt(R**2 + C**2)
   wild = (wild - 10) // 5
   wild = np.clip(wild, -1, 99)

   return wild

class Status:
   def __init__(self, config):
      self.config = config
      self.wilderness = -1
      self.immune = 0
      self.freeze = 0

   def update(self, ent, world, actions):
      self.immune = max(0, self.immune-1)
      self.freeze = max(0, self.freeze-1)
   
      self.wilderness = wilderness(self.config, ent.base.pos)

   def packet(self):
      data = {}
      data['wilderness'] = self.wilderness
      data['immune']     = self.immune
      data['freeze']     = self.freeze
      return data

class Player():
   SERIAL = 0
   def __init__(self, config, iden, pop, name='', color=None):
      self.config = config
      self.repr   = None

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

   #Note: does not stack damage, but still applies to health
   def applyDamage(self, dmg, style):
      self.resources.food.val  = max(0, self.resources.food.val + dmg)
      self.resources.water.val = max(0, self.resources.water.val + dmg)

      self.skills.applyDamage(dmg, style)
      
   def receiveDamage(self, dmg):
      self.resources.health.val = max(0, self.resources.health.val - dmg)
      self.resources.food.val   = max(0, self.resources.food.val - dmg)
      self.resources.water.val  = max(0, self.resources.water.val - dmg)

      self.history.damage = dmg
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
      args = args.values()
      atn.call(world, self, *args)


