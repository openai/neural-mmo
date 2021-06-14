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
        self.target    = None

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

class Combat(Forage):
    def __call__(self, obs):
        actions = super().__call__(obs)
        config  = self.config

        targID, dist = attack.closestTarget(config, self.ob)
        if targID is None:
           return actions

        style = None
        if dist <= config.COMBAT_MELEE_REACH:
            style = Action.Melee
        elif dist <= config.COMBAT_RANGE_REACH:
            style = Action.Range
        elif dist <= config.COMBAT_MAGE_REACH:
            style = Action.Mage

        if not style:
           return actions

        actions[Action.Attack] = {
              Action.Style: style,
              Action.Target: targID}

        return actions
