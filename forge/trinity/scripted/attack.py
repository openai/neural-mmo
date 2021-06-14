from pdb import set_trace as T
import numpy as np

from forge.blade.io.stimulus.static import Stimulus

from forge.trinity.scripted import io, utils

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

   targID = io.Observation.attribute(closestAgent, Entity.ID)
   return int(targID), shortestDist
