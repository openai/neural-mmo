from pdb import set_trace as T

from forge.blade.systems.ai import behavior, move, attack, utils
from forge.blade.io.action import static as Action
from forge.blade.systems import combat


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


def baseline(realm, entity):
   behavior.update(entity)
   actions = {}

   if not entity.attacker:
      min_level = min(combat.level(entity.skills) * 0.5 + 7,
                      15)  # level 3 fighter need 9 food, 9 water, 9 health to fight
      if entity.resources.food <= min_level or entity.resources.water <= min_level or entity.resources.health <= min_level:
         behavior.forage(realm, actions, entity)
      else:
         if not entity.target:
            entity.target = utils.closestTarget(entity, realm.map.tiles,
                                                rng=entity.vision)

         if entity.target \
                 and ((combat.level(entity.target.skills) <= combat.level(
            entity.skills) <= 5)
                      or (combat.level(entity.skills) >= combat.level(
                    entity.target.skills) + 3)):
            behavior.hunt(realm, actions, entity)
         else:
            behavior.forage(realm, actions, entity)
   else:
      entity.target = entity.attacker
      behavior.hunt(realm, actions, entity)

   # behavior.forage(realm, actions, entity)

   return actions
