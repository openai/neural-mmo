from pdb import set_trace as T
import numpy as np

def center(x, mmax):
   return (x - mmax/2.0)/mmax

def oneHot(v, n):
   ary = np.zeros(n)
   ary[v] = 1
   return ary

def entity(ent, other, config):
   r, c = ent.pos
   r = center(r, ent.R)
   c = center(c, ent.C)
   return stats(ent, other, config)

def deltaOneHot(pos, stimSz):
   n = 2 * stimSz + 1
   r, c = pos
   r, c = r+stimSz, c+stimSz
   r = r * n
   idx = int(r + c)
   return oneHot(idx, int(n**2))

def stats(ent, other, config):
   health = ent.health.center()
   food = ent.food.center()
   water = ent.water.center()

   lifetime = center(ent.timeAlive, 100)
   damage = ent.damage
   if damage is None:
      damage = 0
   damage = center(damage, 5)
   freeze = center(ent.freeze, 2)

   #Cant use one hot colors because it causes dimensional
   #conflicts when new pops are added at test time
   sameColor = float(ent.colorInd == other.colorInd) - 0.5

   rSelf, cSelf   = ent.pos
   r, c = center(rSelf, ent.R), center(cSelf, ent.C)

   rOther, cOther = other.pos
   rDelta, cDelta = rOther - rSelf, cOther - cSelf
   ret = np.array([lifetime, health, food, water, 
         r, c, rDelta, cDelta, damage, sameColor, freeze])
   return ret

def environment(env, ent, sz, config):
   R, C = env.shape
   conv, ents =  np.zeros((2, R, C)), []
   for r in range(R):
      for c in range(C):
         t, e = tile(ent, env[r, c], sz, config)
         conv[:, r, c] = t
         ents += e
   assert len(ents) > 0
   ents = np.array(ents)
   return conv, ents

def tile(ent, t, sz, config):
   nTiles = 8
   index = t.state.index
   assert index >= 0 and index < nTiles
   conv = [index, t.nEnts]

   ents = []
   r, c = ent.pos
   for e in t.ents.values():
      statStim = stats(ent, e, config)
      e.stim = statStim
      ents.append(statStim)

   return conv, ents


