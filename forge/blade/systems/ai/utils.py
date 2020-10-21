from pdb import set_trace as T
import numpy as np

from forge.blade.lib.utils import inBounds
from forge.blade.systems import combat
from queue import PriorityQueue, Queue

def validTarget(ent, targ, rng):
   if targ is None or not targ.alive:
      return False
   if lInf(ent, targ) > rng:
      return False
   wild = min(ent.status.wilderness, targ.status.wilderness)
   if abs(combat.level(ent.skills) - combat.level(targ.skills)) > wild:
      return False
   return True

def validResource(ent, tile, rng):
   return tile is not None and tile.state.tex in ('forest', 'water') and distance(ent, tile) <= rng

def directionTowards(ent, targ):
   sr, sc = ent.base.pos
   tr, tc = targ.base.pos

   if abs(sc - tc) > abs(sr - tr):
      direction = (0, np.sign(tc - sc))
   else:
      direction = (np.sign(tr - sr), 0)

   return direction

def closestTarget(ent, tiles, rng=1):
   sr, sc = ent.base.pos
   for d in range(rng+1):
      for r in range(-d, d+1):
         for e in tiles[sr+r, sc-d].ents.values():
            if e is not ent and validTarget(ent, e, rng): return e

         for e in tiles[sr+r, sc+d].ents.values():
            if e is not ent and validTarget(ent, e, rng): return e

         for e in tiles[sr-d, sc+r].ents.values():
            if e is not ent and validTarget(ent, e, rng): return e

         for e in tiles[sr+d, sc+r].ents.values():
            if e is not ent and validTarget(ent, e, rng): return e

def closestResources(ent, tiles, rng=1):
   sr, sc      = ent.pos
   food, water = None, None
   for d in range(rng+1):
      for r in range(-d, d+1):
         if food is None and tiles[sr+r, sc-d].state.tex == 'forest':
            food  = tiles[sr+r, sc-d]
         if water is None and tiles[sr+r, sc-d].state.tex == 'water':
            water = tiles[sr+r, sc-d]

         if food is None and tiles[sr+r, sc+d].state.tex == 'forest':
            food  = tiles[sr+r, sc+d]
         if water is None and tiles[sr+r, sc+d].state.tex == 'water':
            water = tiles[sr+r, sc+d]

         if food is None and tiles[sr-d, sc+r].state.tex == 'forest':
            food  = tiles[sr-d, sc+r]
         if water is None and tiles[sr-d, sc+r].state.tex == 'water':
            water = tiles[sr-d, sc+r]

         if food is None and tiles[sr+d, sc+r].state.tex == 'forest':
            food  = tiles[sr+d, sc+r]
         if water is None and tiles[sr+d, sc+r].state.tex == 'water':
            water = tiles[sr+d, sc+r]

   return food, water



def distance(ent, targ):
   return l1(ent.pos, targ.pos)

def lInf(ent, targ):
   sr, sc = ent.pos
   gr, gc = targ.pos
   return abs(gr-sr) + abs(gc-sc)

def adjacentPos(pos):
   r, c = pos
   return [(r-1, c), (r, c-1), (r+1, c), (r, c+1)]

def forage(tiles, entity, cutoff=100):
   start = entity.pos

   queue = Queue()
   queue.put(start)

   backtrace = {start: None}
   reward    = {start: (entity.resources.food._val, entity.resources.water._val)}
   best      = -1000 
   goal      = start

   while not queue.empty():
      cutoff -= 1
      if cutoff <= 0:
         while goal in backtrace and backtrace[goal] != start:
            goal = backtrace[goal]

         sr, sc = start
         gr, gc = goal 

         return (gr-sr, gc-sc)

      cur = queue.get()

      for nxt in adjacentPos(cur):
         if nxt in backtrace:
            continue

         if not tiles[nxt].habitable:
            continue

         if not inBounds(*nxt, tiles.shape):
            continue

         food, water = reward[cur]
         food  = max(0, food-1)
         water = max(0, water-1)

         if tiles[nxt].state.tex == 'forest':
            food = min(food + entity.resources.food._max//2, entity.resources.food._max) 
         for pos in adjacentPos(nxt):
            if tiles[pos].state.tex == 'water':
               water = min(water + entity.resources.water._max//2, entity.resources.water._max) 
               break

         reward[nxt] = (food, water)

         total = min(food, water)
         if total > best or (total == best and max(food, water) > max(reward[goal])):
            best = total
            goal = nxt 

         queue.put(nxt)
         backtrace[nxt] = cur

#A* Search
def l1(start, goal):
   sr, sc = start
   gr, gc = goal
   return abs(gr-sr) + abs(gc-sc)

def aStar(tiles, start, goal, cutoff=100):
   if start == goal:
      return (0, 0)

   pq = PriorityQueue()
   pq.put((0, start))

   backtrace        = {}
   cost             = {start : 0}

   closestPos       = start
   closestHeuristic = l1(start, goal)
   closestCost      = closestHeuristic

   while not pq.empty():
      #Use approximate solution if budget exhausted
      cutoff -= 1
      if cutoff <= 0:
         if goal not in backtrace:
            goal = closestPos
         break
         
      priority, cur = pq.get()

      if cur == goal:
         break

      for nxt in adjacentPos(cur):
         if not inBounds(*nxt, tiles.shape):
            continue
         if tiles[nxt].impassible:
            continue

         newCost = cost[cur] + 1
         if nxt not in cost or newCost < cost[nxt]:
            cost[nxt] = newCost
            heuristic = l1(goal, nxt)
            priority  = newCost + heuristic

            #Compute approximate solution
            if heuristic < closestHeuristic or (heuristic == closestHeuristic and priority < closestCost):
               closestPos       = nxt
               closestHeuristic = heuristic
               closestCost      = priority

            pq.put((priority, nxt))
            backtrace[nxt] = cur

   while goal in backtrace and backtrace[goal] != start:
      goal = backtrace[goal]

   sr, sc = start
   gr, gc = goal 

   return (gr-sr, gc-sc)
  
#End A*

#Adjacency functions
def adjacentTiles(tiles, ent):
   r, c = ent.base.pos

def adjacentDeltas():
   return [(-1, 0), (1, 0), (0, 1), (0, -1)]

def l1Deltas(s):
   rets = []
   for r in range(-s, s+1):
      for c in range(-s, s+1):
         rets.append((r, c))
   return rets

def posSum(pos1, pos2):
   return pos1[0] + pos2[0], pos1[1] + pos2[1]

def adjacentEmptyPos(env, pos):
   return [p for p in adjacentPos(pos)
         if inBounds(*p, env.size)]

def adjacentTiles(env, pos):
   return [env.tiles[p] for p in adjacentPos(pos)
         if inBounds(*p, env.size)]

def adjacentMats(env, pos):
   return [type(env.tiles[p].mat) for p in adjacentPos(pos)
         if inBounds(*p, env.shape)]

def adjacencyDelMatPairs(env, pos):
   return zip(adjacentDeltas(), adjacentMats(env, pos))
###End###

