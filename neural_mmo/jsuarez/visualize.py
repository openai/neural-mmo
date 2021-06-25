###Old visualization code. TODO: Update for the new client

#These hooks are outdated. Better policy
#visualization for the new client is planned
#for a future update
def visDeps(self):
   from forge.blade.core import realm
   from forge.blade.core.tile import Tile
   colorInd = int(self.config.NPOP*np.random.rand())
   color    = Neon.color12()[colorInd]
   color    = (colorInd, color)
   ent = realm.Desciple(-1, self.config, color).server
   targ = realm.Desciple(-1, self.config, color).server

   sz = 15
   tiles = np.zeros((sz, sz), dtype=object)
   for r in range(sz):
      for c in range(sz):
         tiles[r, c] = Tile(enums.Grass, r, c, 1, None)

   targ.pos = (7, 7)
   tiles[7, 7].addEnt(0, targ)
   posList, vals = [], []
   for r in range(sz):
      for c in range(sz):
         ent.pos  = (r, c)
         tiles[r, c].addEnt(1, ent)
         #_, _, val = self.net(tiles, ent)
         val = np.random.rand()
         vals.append(float(val))
         tiles[r, c].delEnt(1)
         posList.append((r, c))
   vals = list(zip(posList, vals))
   return vals

#These hooks are outdated. Better policy
#visualization for the new client is planned
#for a future update
def visVals(self, food='max', water='max'):
   from forge.blade.core import realm
   posList, vals = [], []
   R, C = self.world.shape
   for r in range(self.config.BORDER, R-self.config.BORDER):
       for c in range(self.config.BORDER, C-self.config.BORDER):
         colorInd = int(self.config.NPOP*np.random.rand())
         color    = Neon.color12()[colorInd]
         color    = (colorInd, color)
         ent = entity.Player(-1, color, self.config)
         ent._r.update(r)
         ent._c.update(c)
         if food != 'max':
            ent._food = food
         if water != 'max':
            ent._water = water
         posList.append(ent.pos)

         self.world.env.tiles[r, c].addEnt(ent.entID, ent)
         stim = self.world.env.stim(ent.pos, self.config.STIM)
         #_, _, val = self.net(stim, ent)
         val = np.random.rand()
         self.world.env.tiles[r, c].delEnt(ent.entID)
         vals.append(float(val))

   vals = list(zip(posList, vals))
   return vals

