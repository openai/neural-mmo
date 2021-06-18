from pdb import set_trace as T

from forge.trinity.scripted import behavior, move, attack, utils, io
from forge.blade.io.stimulus.static import Stimulus
from forge.blade.io.action import static as Action

class Scripted:
    def __init__(self, config):
        self.config    = config

        self.food_max  = 0
        self.water_max = 0

        self.spawnR    = None
        self.spawnC    = None

    @property
    def forage_criterion(self):
        min_level = 7
        return self.food <= min_level or self.water <= min_level

    def forage(self):
        move.forageDijkstra(self.config, self.ob, self.actions, self.food_max, self.water_max)

    def explore(self):
        move.explore(self.config, self.ob, self.actions, self.spawnR, self.spawnC)

    @property
    def downtime(self):
        return not self.forage_criterion and self.attacker is None

    def evade(self):
        move.evade(self.config, self.ob, self.actions, self.attacker)
        self.target     = self.attacker
        self.targetID   = self.attackerID
        self.targetDist = self.attackerDist

    def attack(self):
        if self.target is not None:
           assert self.targetID is not None
           attack.target(self.config, self.actions, self.style, self.targetID)

    def select_combat_style(self):
       if self.target is None:
          return

       if self.targetDist <= self.config.COMBAT_MELEE_REACH:
          self.style = Action.Melee
       elif self.targetDist <= self.config.COMBAT_RANGE_REACH:
          self.style = Action.Range
       else:
          self.style = Action.Mage

    def target_weak(self):
        if self.closest is None:
            return False

        selfLevel = io.Observation.attribute(self.ob.agent, Stimulus.Entity.Level)
        targLevel = io.Observation.attribute(self.closest, Stimulus.Entity.Level)
        
        if targLevel <= selfLevel <= 5 or selfLevel >= targLevel + 3:
           self.target     = self.closest
           self.targetID   = self.closestID
           self.targetDist = self.closestDist

    def scan_agents(self):
        self.closest, self.closestDist   = attack.closestTarget(self.config, self.ob)
        self.attacker, self.attackerDist = attack.attacker(self.config, self.ob)

        self.closestID = None
        if self.closest is not None:
           self.closestID = io.Observation.attribute(self.closest, Stimulus.Entity.ID)

        self.attackerID = None
        if self.attacker is not None:
           self.attackerID = io.Observation.attribute(self.attacker, Stimulus.Entity.ID)

        self.style      = None
        self.target     = None
        self.targetID   = None
        self.targetDist = None

    def adaptive_control_and_targeting(self, explore=True):
        self.scan_agents()

        if self.attacker is not None:
           self.evade()
           return

        if self.forage_criterion or not explore:
           self.forage()
        else:
           self.explore()

        self.target_weak()

    def __call__(self, obs):
        self.actions = {}

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

class Random(Scripted):
    def __call__(self, obs):
        super().__call__(obs)

        move.random(self.config, self.ob, self.actions)
        return self.actions

class Meander(Scripted):
    def __call__(self, obs):
        super().__call__(obs)

        move.meander(self.config, self.ob, self.actions)
        return self.actions

class ForageNoExplore(Scripted):
    def __call__(self, obs):
        super().__call__(obs)

        self.forage()

        return self.actions

class Forage(Scripted):
    def __call__(self, obs):
        super().__call__(obs)

        if self.forage_criterion:
           self.forage()
        else:
           self.explore()

        return self.actions

class CombatNoExplore(Scripted):
    def __call__(self, obs):
        super().__call__(obs)

        self.adaptive_control_and_targeting(explore=False)

        self.style = Action.Range
        self.attack()

        return self.actions
 
class Combat(Scripted):
    def __call__(self, obs):
        super().__call__(obs)

        self.adaptive_control_and_targeting()

        self.style = Action.Range
        self.attack()

        return self.actions

class CombatTribrid(Scripted):
    def __call__(self, obs):
        super().__call__(obs)

        self.adaptive_control_and_targeting()

        self.select_combat_style()
        self.attack()

        return self.actions
