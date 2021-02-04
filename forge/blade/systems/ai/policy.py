from pdb import set_trace as T

from forge.blade.systems.ai import behavior, move, attack, utils
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

def forage(realm, entity, explore=True, forage=behavior.forageDijkstra):
   return baseline(realm, entity, explore, forage, combat=False)

def combat(realm, entity, explore=True, forage=behavior.forageDijkstra):
   return baseline(realm, entity, explore, forage, combat=True)

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
