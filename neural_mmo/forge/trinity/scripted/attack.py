from pdb import set_trace as T
import numpy as np

from neural_mmo.forge.blade.io.stimulus.static import Stimulus
from neural_mmo.forge.blade.io.action import static as Action

from neural_mmo.forge.trinity.scripted import io, utils

def closestTarget(config, ob):
   shortestDist = np.inf
   closestAgent = None

   Entity = Stimulus.Entity
   agent  = ob.agent

   sr = io.Observation.attribute(agent, Entity.R)
   sc = io.Observation.attribute(agent, Entity.C)
   start = (sr, sc)

   for target in ob.agents:
      exists = io.Observation.attribute(target, Entity.Self)
      if not exists:
         continue

      tr = io.Observation.attribute(target, Entity.R)
      tc = io.Observation.attribute(target, Entity.C)

      goal = (tr, tc)
      dist = utils.l1(start, goal)

      if dist < shortestDist and dist != 0:
          shortestDist = dist
          closestAgent = target

   if closestAgent is None:
      return None, None

   return closestAgent, shortestDist

def attacker(config, ob):
   Entity = Stimulus.Entity

   sr = io.Observation.attribute(ob.agent, Entity.R)
   sc = io.Observation.attribute(ob.agent, Entity.C)
 
   attackerID = io.Observation.attribute(ob.agent, Entity.AttackerID)

   if attackerID == 0:
       return None, None

   for target in ob.agents:
      identity = io.Observation.attribute(target, Entity.ID)
      if identity == attackerID:
         tr = io.Observation.attribute(target, Entity.R)
         tc = io.Observation.attribute(target, Entity.C)
         dist = utils.l1((sr, sc), (tr, tc))
         return target, dist
   return None, None

def target(config, actions, style, targetID):
   actions[Action.Attack] = {
         Action.Style: style,
         Action.Target: targetID}

