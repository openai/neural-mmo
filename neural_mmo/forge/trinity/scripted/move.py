from pdb import set_trace as T
import numpy as np
import random as rand

from neural_mmo.forge.blade.lib import material
from neural_mmo.forge.blade.io.stimulus.static import Stimulus
from neural_mmo.forge.blade.io.action import static as Action
from queue import PriorityQueue, Queue

from neural_mmo.forge.trinity.scripted import io, utils

def adjacentPos(pos):
   r, c = pos
   return [(r - 1, c), (r, c - 1), (r + 1, c), (r, c + 1)]

def inSight(dr, dc, vision):
    return (
          dr >= -vision and
          dc >= -vision and
          dr <= vision and
          dc <= vision)

def vacant(tile):
   Tile     = Stimulus.Tile
   occupied = io.Observation.attribute(tile, Tile.NEnts)
   matl     = io.Observation.attribute(tile, Tile.Index)

   return matl in material.Habitable and not occupied

def random(config, ob, actions):
   direction = rand.choice(Action.Direction.edges)
   actions[Action.Move] = {Action.Direction: direction}

def towards(direction):
   if direction == (-1, 0):
      return Action.North
   elif direction == (1, 0):
      return Action.South
   elif direction == (0, -1):
      return Action.West
   elif direction == (0, 1):
      return Action.East
   else:
      return rand.choice(Action.Direction.edges)

def pathfind(config, ob, actions, rr, cc):
   direction = aStar(config, ob, actions, rr, cc)
   direction = towards(direction)
   actions[Action.Move] = {Action.Direction: direction}

def meander(config, ob, actions):
   agent  = ob.agent
   Entity = Stimulus.Entity
   Tile   = Stimulus.Tile

   r = io.Observation.attribute(agent, Entity.R)
   c = io.Observation.attribute(agent, Entity.C)

   cands = []
   if vacant(ob.tile(-1, 0)):
      cands.append((-1, 0))
   if vacant(ob.tile(1, 0)):
      cands.append((1, 0))
   if vacant(ob.tile(0, -1)):
      cands.append((0, -1))
   if vacant(ob.tile(0, 1)):
      cands.append((0, 1))
   if not cands:
      return (-1, 0)

   direction = rand.choices(cands)[0]
   direction = towards(direction)
   actions[Action.Move] = {Action.Direction: direction}

def explore(config, ob, actions, spawnR, spawnC):
   vision = config.NSTIM
   sz     = config.TERRAIN_SIZE
   Entity = Stimulus.Entity
   Tile   = Stimulus.Tile

   agent  = ob.agent
   r      = io.Observation.attribute(agent, Entity.R)
   c      = io.Observation.attribute(agent, Entity.C)

   centR, centC = sz//2, sz//2
   vR, vC = centR-spawnR, centC-spawnC

   mmag = max(abs(vR), abs(vC))
   rr   = int(np.round(vision*vR/mmag))
   cc   = int(np.round(vision*vC/mmag))
   pathfind(config, ob, actions, rr, cc)

def evade(config, ob, actions, attacker):
   Entity = Stimulus.Entity

   sr     = io.Observation.attribute(ob.agent, Entity.R)
   sc     = io.Observation.attribute(ob.agent, Entity.C)

   gr     = io.Observation.attribute(attacker, Entity.R)
   gc     = io.Observation.attribute(attacker, Entity.C)

   rr, cc = (2*sr - gr, 2*sc - gc)

   pathfind(config, ob, actions, rr, cc)

def forageDijkstra(config, ob, actions, food_max, water_max, cutoff=100):
   vision = config.NSTIM
   Entity = Stimulus.Entity
   Tile   = Stimulus.Tile

   agent  = ob.agent
   food   = io.Observation.attribute(agent, Entity.Food)
   water  = io.Observation.attribute(agent, Entity.Water)

   best      = -1000 
   start     = (0, 0)
   goal      = (0, 0)

   reward    = {start: (food, water)}
   backtrace = {start: None}

   queue = Queue()
   queue.put(start)

   while not queue.empty():
      cutoff -= 1
      if cutoff <= 0:
         break

      cur = queue.get()
      for nxt in adjacentPos(cur):
         if nxt in backtrace:
            continue

         if not inSight(*nxt, vision):
            continue

         tile     = ob.tile(*nxt)
         matl     = io.Observation.attribute(tile, Tile.Index)
         occupied = io.Observation.attribute(tile, Tile.NEnts)

         if not vacant(tile):
            continue

         food, water = reward[cur]
         food  = max(0, food - 1)
         water = max(0, water - 1)

         if matl == material.Forest.index:
            food = min(food+food_max//2, food_max)
         for pos in adjacentPos(nxt):
            if not inSight(*pos, vision):
               continue

            tile = ob.tile(*pos)
            matl = io.Observation.attribute(tile, Tile.Index)
 
            if matl == material.Water.index:
               water = min(water+water_max//2, water_max)
               break

         reward[nxt] = (food, water)

         total = min(food, water)
         if total > best or (
                 total == best and max(food, water) > max(reward[goal])):
            best = total
            goal = nxt

         queue.put(nxt)
         backtrace[nxt] = cur

   while goal in backtrace and backtrace[goal] != start:
      goal = backtrace[goal]

   direction = towards(goal)
   actions[Action.Move] = {Action.Direction: direction}

def aStar(config, ob, actions, rr, cc, cutoff=100):
   Entity = Stimulus.Entity
   Tile   = Stimulus.Tile
   vision = config.NSTIM

   start = (0, 0)
   goal  = (rr, cc)

   if start == goal:
      return (0, 0)

   pq = PriorityQueue()
   pq.put((0, start))

   backtrace = {}
   cost = {start: 0}

   closestPos = start
   closestHeuristic = utils.l1(start, goal)
   closestCost = closestHeuristic

   while not pq.empty():
      # Use approximate solution if budget exhausted
      cutoff -= 1
      if cutoff <= 0:
         if goal not in backtrace:
            goal = closestPos
         break

      priority, cur = pq.get()

      if cur == goal:
         break

      for nxt in adjacentPos(cur):
         if not inSight(*nxt, vision):
            continue

         tile     = ob.tile(*nxt)
         matl     = io.Observation.attribute(tile, Tile.Index)
         occupied = io.Observation.attribute(tile, Tile.NEnts)

         #if not vacant(tile):
         #   continue

         if occupied:
            continue

         #Omitted water from the original implementation. Seems key
         if matl in material.Impassible:
            continue

         newCost = cost[cur] + 1
         if nxt not in cost or newCost < cost[nxt]:
            cost[nxt] = newCost
            heuristic = utils.lInfty(goal, nxt)
            priority = newCost + heuristic

            # Compute approximate solution
            if heuristic < closestHeuristic or (
                    heuristic == closestHeuristic and priority < closestCost):
               closestPos = nxt
               closestHeuristic = heuristic
               closestCost = priority

            pq.put((priority, nxt))
            backtrace[nxt] = cur

   #Not needed with scuffed material list above
   #if goal not in backtrace:
   #   goal = closestPos

   while goal in backtrace and backtrace[goal] != start:
      goal = backtrace[goal]

   return goal

