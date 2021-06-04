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
            vars()[attrName] = idx

class Observation:
    def __init__(self, config, obs):
        self.config = config
        self.obs    = obs
        self.delta  = config.NSTIM

        self.tiles  = self.obs['Tile']
        self.agents = self.obs['Entity']
        self.n      = self.agents['N']

    def pos(self, rDelta, cDelta):
        return self.delta + rDelta, self.delta + cDelta

    def tile(self, rDelta, cDelta):
        pos = self.pos(rDelta, cDelta)
        #return self.tiles[*pos]
        
    def agent(self, entID):
        return self.agents[entID]

class Random:
    def __init__(self, config):
        self.config = config

    def __call__(self, obs, state, seq_lens):
        config  = self.config

        obs     = Observation(config, obs)
        actions = defaultdict(lambda: defaultdict(list))

        actions[Action.Move][Action.Direction].append(torch.Tensor([1,0,0,0]))
        actions[Action.Attack][Action.Style].append(torch.Tensor([1,0,0]))

        targ = torch.zeros(config.N_AGENT_OBS)
        targ[1] = 1
        actions[Action.Attack][Action.Target].append(targ)

        for atnKey, atn in actions.items():
            for argKey, args in atn.items():
                actions[atnKey][argKey] = torch.stack(args)

        return actions, state

        #actions[Action.Move] = {Action.Direction: move.habitable(realm.map.tiles, entity)} 

#def forage(realm, entity, explore=True, forage=behavior.forageDijkstra):
#   return baseline(realm, entity, explore, forage, combat=False)

def combat(realm, entity, explore=True, forage=behavior.forageDijkstra):
   return baseline(realm, entity, explore, forage, combat=True)

def random(realm, entity, explore=None, forage=None):
    actions = {}
    behavior.meander(realm, actions, entity)
    return  actions

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
