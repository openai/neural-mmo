from pdb import set_trace as T
import numpy as np

import nmmo
from nmmo.systems.ai import move, attack, utils

def update(entity):
   '''Update validity of tracked entities'''
   if not utils.validTarget(entity, entity.attacker, entity.vision):
      entity.attacker = None
   if not utils.validTarget(entity, entity.target, entity.vision):
      entity.target = None
   if not utils.validTarget(entity, entity.closest, entity.vision):
      entity.closest = None

   if entity.__class__.__name__ != 'Player':
      return

   if not utils.validResource(entity, entity.food, entity.vision):
      entity.food = None
   if not utils.validResource(entity, entity.water, entity.vision):
      entity.water = None
      
def pathfind(realm, actions, entity, target):
   actions[nmmo.action.Move]   = {nmmo.action.Direction: move.pathfind(realm.map.tiles, entity, target)}

def explore(realm, actions, entity):
   sz   = realm.config.TERRAIN_SIZE
   r, c = entity.pos

   spawnR, spawnC = entity.spawnPos
   centR, centC   = sz//2, sz//2

   vR, vC = centR-spawnR, centC-spawnC

   mmag = max(abs(vR), abs(vC))
   rr   = r + int(np.round(entity.vision*vR/mmag))
   cc   = c + int(np.round(entity.vision*vC/mmag))

   tile = realm.map.tiles[rr, cc]
   pathfind(realm, actions, entity, tile)

def explore(config, ob, actions, spawnR, spawnC):
   vision = config.NSTIM
   sz     = config.TERRAIN_SIZE
   Entity = nmmo.Serialized.Entity
   Tile   = nmmo.Serialized.Tile

   agent  = ob.agent
   r      = utils.Observation.attribute(agent, Entity.R)
   c      = utils.Observation.attribute(agent, Entity.C)

   centR, centC   = sz//2, sz//2

   vR, vC = centR-spawnR, centC-spawnC

   mmag = max(abs(vR), abs(vC))
   rr   = int(np.round(vision*vR/mmag))
   cc   = int(np.round(vision*vC/mmag))

   pathfind(config, ob, actions, rr, cc)

def meander(realm, actions, entity):
   actions[nmmo.action.Move] = {nmmo.action.Direction: move.habitable(realm.map.tiles, entity)}

def evade(realm, actions, entity):
   actions[nmmo.action.Move] = {nmmo.action.Direction: move.antipathfind(realm.map.tiles, entity, entity.attacker)}

def hunt(realm, actions, entity):
   #Move args
   distance = utils.distance(entity, entity.target)

   direction = None
   if distance == 0:
      direction = move.random()
   elif distance > 1:
      direction = move.pathfind(realm.map.tiles, entity, entity.target)

   if direction is not None:
      actions[nmmo.action.Move] = {nmmo.action.Direction: direction}

   attack(realm, actions, entity)

def attack(realm, actions, entity):
   distance = utils.lInfty(entity.pos, entity.target.pos)
   if distance > entity.skills.style.attackRange(realm.config):
      return

   actions[nmmo.action.Attack] = {
         nmmo.action.Style: entity.skills.style,
         nmmo.action.Target: entity.target}

