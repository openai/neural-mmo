from pdb import set_trace as T

from forge.trinity.scripted import behavior, move, attack, utils, io
from forge.blade.io.stimulus.static import Stimulus
from forge.blade.io.action import static as Action

class Base:
    def __init__(self, config):
        self.config    = config

        self.food_max  = 0
        self.water_max = 0

        self.spawnR    = None
        self.spawnC    = None

    def __call__(self, obs):
        self.ob = io.Observation(self.config, obs)
        agent   = self.ob.agent

        self.food   = io.Observation.attribute(agent, Stimulus.Entity.Food)
        self.water  = io.Observation.attribute(agent, Stimulus.Entity.Water)

        if self.food > self.food_max:
           self.food_max = self.food
        if self.water > self.water_max:
           self.water_max = self.water

        if self.spawnR is None:
            self.spawnR = io.Observation.attribute(agent, Stimulus.Entity.R)
        if self.spawnC is None:
            self.spawnC = io.Observation.attribute(agent, Stimulus.Entity.C)

class Random(Base):
    def __call__(self, obs):
        super().__call__(obs)
        actions = {}

        move.random(self.config, self.ob, actions)
        return actions

class Meander(Base):
    def __call__(self, obs):
        super().__call__(obs)
        actions = {}

        move.meander(self.config, self.ob, actions)
        return actions

class ForageNoExplore(Base):
    def __call__(self, obs):
        super().__call__(obs)
        config  = self.config
        actions = {}

        move.forageDijkstra(config, self.ob, actions, self.food_max, self.water_max)
        return actions

class Forage(Base):
    def __call__(self, obs):
        super().__call__(obs)
        config  = self.config
        actions = {}

        min_level = 7
        if (self.food <= min_level or self.water <= min_level):
           move.forageDijkstra(config, self.ob, actions, self.food_max, self.water_max)
        else:
           move.explore(config, self.ob, actions, self.spawnR, self.spawnC)

        return actions

class CombatBase(Base):
   def __call__(self, obs):
        super().__call__(obs)
        config  = self.config

        agent                            = self.ob.agent
        self.target, self.targetDist     = attack.closestTarget(config, self.ob)
        self.attacker, self.attackerDist = attack.attacker(config, self.ob)

        self.targetID = None
        if self.target is not None:
           self.targetID   = io.Observation.attribute(self.target, Stimulus.Entity.ID)

        self.attackerID = None
        if self.attacker is not None:
           self.attackerID = io.Observation.attribute(self.attacker, Stimulus.Entity.ID)
 
class CombatNoExplore(CombatBase):
    def __call__(self, obs):
        super().__call__(obs)
        agent   = self.ob.agent
        config  = self.config
        actions = {}

        argumentID = None
        min_level  = 7
        if (self.food > min_level and self.water > min_level and self.attacker is not None):
           move.evade(config, self.ob, actions, self.attacker)
           argumentID = self.attackerID
        else:
           move.forageDijkstra(config, self.ob, actions, self.food_max, self.water_max)
           if self.target is not None:
              selfLevel = io.Observation.attribute(agent, Stimulus.Entity.Level)
              targLevel = io.Observation.attribute(self.target, Stimulus.Entity.Level)
              if targLevel <= selfLevel <= 5 or selfLevel >= targLevel + 3:
                  argumentID = self.targetID

        if argumentID is not None: 
           actions[Action.Attack] = {
                 Action.Style: Action.Range,
                 Action.Target: argumentID}

        return actions
 
class Combat(CombatBase):
    def __call__(self, obs):
        super().__call__(obs)
        agent   = self.ob.agent
        config  = self.config
        actions = {}

        argumentID = None
        min_level  = 7
        if (self.food <= min_level or self.water <= min_level):
           move.forageDijkstra(config, self.ob, actions, self.food_max, self.water_max)
        elif self.attacker is not None:
           move.evade(config, self.ob, actions, self.attacker)
           argumentID = self.attackerID
        elif self.target is not None:
           move.explore(config, self.ob, actions, self.spawnR, self.spawnC)
           selfLevel = io.Observation.attribute(agent, Stimulus.Entity.Level)
           targLevel = io.Observation.attribute(self.target, Stimulus.Entity.Level)
           if targLevel <= selfLevel <= 5 or selfLevel >= targLevel + 3:
              argumentID = self.targetID 
        else:
           move.explore(config, self.ob, actions, self.spawnR, self.spawnC)

        if argumentID is not None: 
           actions[Action.Attack] = {
                 Action.Style: Action.Range,
                 Action.Target: argumentID}

        return actions

class CombatTribrid(CombatBase):
    def __call__(self, obs):
        super().__call__(obs)
        agent   = self.ob.agent
        config  = self.config
        actions = {}

        argumentID = None
        min_level  = 7
        if (self.food <= min_level or self.water <= min_level):
           move.forageDijkstra(config, self.ob, actions, self.food_max, self.water_max)
        elif self.attacker is not None:
           move.evade(config, self.ob, actions, self.attacker)
           self.dist  = self.attackerDist
           argumentID = self.attackerID
        elif self.target is not None:
           move.explore(config, self.ob, actions, self.spawnR, self.spawnC)
           selfLevel = io.Observation.attribute(agent, Stimulus.Entity.Level)
           targLevel = io.Observation.attribute(self.target, Stimulus.Entity.Level)
           if targLevel <= selfLevel <= 5 or selfLevel >= targLevel + 3:
              self.dist  = self.targetDist
              argumentID = self.targetID 
        else:
           move.explore(config, self.ob, actions, self.spawnR, self.spawnC)

        if argumentID is not None: 
           style = None
           if self.dist <= config.COMBAT_MELEE_REACH:
              style = Action.Melee
           elif self.dist <= config.COMBAT_RANGE_REACH:
              style = Action.Range
           else:
              style = Action.Mage
 
           actions[Action.Attack] = {
                 Action.Style: style,
                 Action.Target: argumentID}

        return actions
