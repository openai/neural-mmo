from pdb import set_trace as T

import torch
from collections import defaultdict

from forge.blade.systems.ai import behavior, move, attack, utils
from forge.blade.io.stimulus.static import Stimulus
from forge.blade.io.action import static as Action
from forge.blade import systems

def passive(realm, entity):
   behavior.update(entity)
   actions = {}

   behavior.meander(realm, actions, entity)

   return actions

def neutral(realm, entity):
   behavior.update(entity)
   actions = {}

   if not entity.attacker:
      behavior.meander(realm, actions, entity)
   else:
      entity.target = entity.attacker
      behavior.hunt(realm, actions, entity)

   return actions

def hostile(realm, entity):
   behavior.update(entity)
   actions = {}

   # This is probably slow
   if not entity.target:
      entity.target = utils.closestTarget(entity, realm.map.tiles,
                                          rng=entity.vision)

   if not entity.target:
      behavior.meander(realm, actions, entity)
   else:
      behavior.hunt(realm, actions, entity)

   return actions

class Attributes:
    for objName, obj in Stimulus:
        for idx, (attrName, attr) in enumerate(obj):
            attr.index = idx
            #Indexing error
            #vars()[attrName] = idx

class Random:
    def __init__(self, config):
        self.config    = config
        self.food_max  = 0
        self.water_max = 0
        self.spawnR    = None
        self.spawnC    = None
        self.target    = None

    def __call__(self, obs):
        config  = self.config
        #agentID = ob.agentID

        actions = {}
        ob      = utils.Observation(config, obs)

        agent  = ob.agent
        food   = utils.Observation.attribute(agent, Stimulus.Entity.Food)
        water  = utils.Observation.attribute(agent, Stimulus.Entity.Water)

        if food > self.food_max:
           self.food_max = food
        if water > self.water_max:
           self.water_max = water

        if self.spawnR is None:
            self.spawnR = utils.Observation.attribute(agent, Stimulus.Entity.R)
        if self.spawnC is None:
            self.spawnC = utils.Observation.attribute(agent, Stimulus.Entity.C)

        min_level = 7
        if (food <= min_level or water <= min_level):
           behavior.forageDijkstra(config, ob, actions, self.food_max, self.water_max)
        else:
           behavior.explore(config, ob, actions, self.spawnR, self.spawnC)

        targID, dist = utils.closestTarget(config, ob)
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


        '''
        elif entity.attacker and combat:
           entity.target = entity.attacker
           behavior.evade(realm, actions, entity)
           behavior.attack(realm, actions, entity)
        elif entity.target and combat:
           downtime(realm, actions, entity)
           entLvl  = systems.combat.level(entity.skills)
           targLvl = systems.combat.level(entity.target.skills)
        if targLvl <=  entLvl <= 5 or entLvl >= targLvl+3:
           behavior.attack(realm, actions, entity)
        else:
           downtime(realm, actions, entity)
        '''

 
        #behavior.forageDijkstra(config, ob, actions, self.food_max, self.water_max)
        behavior.explore(config, ob, actions, self.spawnR, self.spawnC)

        #actions[Action.Attack][Action.Style].append(torch.Tensor([1,0,0]))
        #actions[Action.Attack][Action.Style].append(torch.Tensor([1,0,0]))

        return actions

        #actions[Action.Move] = {Action.Direction: move.habitable(realm.map.tiles, entity)} 


def forage(realm, entity, explore=True, forage=behavior.forageDijkstra):
   return baseline(realm, entity, explore, forage, combat=False)

def combat(realm, entity, explore=True, forage=behavior.forageDijkstra):
   return baseline(realm, entity, explore, forage, combat=True)

#def random(realm, entity, explore=None, forage=None):
#    actions = {}
#    behavior.meander(realm, actions, entity)
#    return  actions

def baseline(realm, entity, explore, forage, combat):
   behavior.update(entity)
   actions = {}

   #Baseline only considers nearest entity
   entity.target = utils.closestTarget(entity,
         realm.map.tiles, rng=entity.vision)

   #Define agent behavior during downtime
   if explore:
      downtime = behavior.explore
   else:
      downtime = forage

   #Forage if low on resources
   min_level = 7
   if (entity.resources.food <= min_level
         or entity.resources.water <= min_level):
      forage(realm, actions, entity)
   elif entity.attacker and combat:
      entity.target = entity.attacker
      behavior.evade(realm, actions, entity)
      behavior.attack(realm, actions, entity)
   elif entity.target and combat:
      downtime(realm, actions, entity)
      entLvl  = systems.combat.level(entity.skills)
      targLvl = systems.combat.level(entity.target.skills)
      if targLvl <=  entLvl <= 5 or entLvl >= targLvl+3:
         behavior.attack(realm, actions, entity)
   else:
      downtime(realm, actions, entity)

   return actions
