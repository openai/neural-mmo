from pdb import set_trace as T
import numpy as np
import time
from scipy.misc import imresize, imsave
from enum import Enum

import pygame
from pygame import Surface

from forge.embyr import embyr
from forge.embyr import utils as renderutils
from forge.embyr import render
from forge.embyr.texture import TextureInitializer
from forge.blade.lib.enums import Neon, Color256, Defaults
from forge.blade.action.v2 import Attack
from forge.blade.action import action

names = open('resource/ents.txt').read().splitlines()
NANIM = 8 
class Map(embyr.Container):
   def __init__(self, realm, textures, 
         canvasSize, tileSz, mipLevels, iso, **kwargs):
      super().__init__(canvasSize, **kwargs)
      self.realm, self.textures, self.iso = realm, textures, iso
      self.tileSz, self.mipLevels  = tileSz, mipLevels

   def render(self, trans):
      self.reset()
      envCrop, txSz = embyr.mapCrop(self, 
            self.realm.world.env.inds(), self.tileSz, trans)
      txSz = embyr.mipKey(txSz, self.mipLevels)
      embyr.renderMap(self, envCrop, txSz, self.textures.material[txSz], self.iso)
      self.flip()  
      return self.surf

   def refresh(self, trans, iso):
      self.iso = iso
      return self.render(trans)

class SpriteGroup:
   def __init__(self, tex, fonts):
      self.tex, self.fonts = tex, fonts
      self.sprites = []

   def add(self, ent):
      sprite = Sprite(ent)
      self.sprites.append(sprite)

   def remove(self, ent):
      self.sprites.remove(ent)

   def update(self):
      for e in self.sprites:
         e.update()

   def render(self, screen, offsets, txSz, iso):
      for e in self.sprites:
         #crop
         e.render(screen, offsets, self.tex, self.fonts, txSz, iso)

class Sprite:
   def __init__(self, ent):
      self.ent = ent
      self.frame = 0
 
      #self.pos = ent.lastPos
      #self.newPos = ent.pos

      self.lastPos = ent.lastPos
      if not ent.alive:
         self.lastPos = ent.pos
      r, c = self.lastPos
      self.posAnim = self.lastPos
      rNew, cNew = self.ent.pos

      self.rOff, self.cOff = np.sign(rNew-r), np.sign(cNew-c)
      #self.targ = ent.attack.args
      nNames = len(names)
      self.entName = names[int(ent.entID)%nNames]
      self.nameColor = Neon.color12()[int(ent.entID) % 12]
       
   def update(self):
      self.frame += 1.0/NANIM
      
      r, c = self.lastPos
      rAnim = r + self.frame * self.rOff
      cAnim = c + self.frame * self.cOff
      self.posAnim = (rAnim, cAnim)

   def render(self, screen, offsets, tex, fonts, txSz, iso):
      self.targ = self.ent.targ
      render.renderSprite(screen, self, offsets, tex, fonts, txSz, iso) 
class Ent(embyr.Container):
   def __init__(self, realm, fonts, textures, 
         canvasSize, tileSz, mipLevels, iso, **kwargs):
      super().__init__(canvasSize, **kwargs)
      self.realm, self.fonts, self.textures = realm, fonts, textures
      self.tileSz, self.mipLevels, self.iso = tileSz, mipLevels, iso

   def render(self, trans, keyframe):
      tiles = self.realm.world.env.tiles
      if keyframe:
         self.makeSprites(tiles)
      self.sprites.update()
      self.renderEnts(trans)
      return self.surf

   def renderEnts(self, trans):
      self.reset()
      R, C = self.realm.world.env.shape
      offs, txSz = embyr.offsets(self, R, C, self.tileSz, trans)
      txSz = embyr.mipKey(txSz, self.mipLevels)
      minH, _, minW, _ = offs
      self.sprites.render(self, (minH, minW), txSz, self.iso)
      return self.surf
 
   def makeSprites(self, tiles):
      self.sprites = SpriteGroup(self.textures, self.fonts)
      for tile in tiles.ravel():
         for e in tile.ents.values():
            self.sprites.add(e)
 
   def refresh(self, trans, iso):
      self.iso = iso
      return self.render(None, trans, keyframe=False)

class EnvViewport(embyr.Container):
   def __init__(self, realm, textures, 
         mipLevels, fonts, canvasSize, iso, **kwargs):
      super().__init__(canvasSize, **kwargs)
      self.textures, self.mipLevels, self.fonts = textures, mipLevels, fonts
      self.tileSz, self.iso = textures.tileSz, iso

      self.map = Map(realm, self.textures, 
            canvasSize, self.tileSz, mipLevels, iso)
      self.ent = Ent(realm, fonts, self.textures, canvasSize, 
            self.tileSz, mipLevels, iso, key=Neon.BLACK.rgb)
      self.screenshot_idx = 0

   def render(self, trans, keyframe=True):
      mmap = self.map.render(trans)
      ent  = self.ent.render(trans, keyframe)
      self.reset()

      self.blit(mmap, (0, 0))
      self.blit(ent, (0, 0))

      self.flip()
      return self.surf

   def refresh(self, trans, iso):
      self.iso = iso
      mmap = self.map.refresh(trans, self.iso)
      ent  = self.ent.refresh(trans, self.iso)

      self.blit(mmap, (0, 0))
      self.blit(ent, (0, 0))

      self.flip()
      return self.surf

   def leftScreenshot(self):
      print("Saved left screenshot")
      im = pygame.surfarray.array3d(self.surf)
      im = im.transpose((1, 0, 2))
      imsave("env{}.png".format(self.screenshot_idx), im)
      self.screenshot_idx += 1

class DatViewport(embyr.Container):
   def __init__(self, realm, textures, fonts, 
         canvasSize, conf, **kwargs):
      super().__init__(canvasSize, **kwargs)
      self.W, self.H = canvasSize
      self.graphHeigh = 128
      self.exchangeWidth = 128
      self.conf = conf
      self.screenshot_idx = 0

      self.idx = 0
      self.nIdx = 3
      self.textures = textures
      self.counts = Counts(realm, textures.mipLevels, 
              canvasSize, border=3)
      self.deps   = Deps(realm, 
              canvasSize, border=3)
      self.action = Action(realm, 
              canvasSize, border=3)
      self.vals   = Vals(realm, textures.mipLevels, canvasSize, 
            conf, border=3)
      self.help   = Help(fonts, canvasSize, border=3)
      self.transparent = False
      self.charIdx = {'h':0, 'v':1, 'c':2, 't':3, 'f':4, 
              'w':5, 'q':6, 'a':7, '1':8, '2':9, 
              '3':10, 'd': 11, ']':12}
      self.funcIdx = {}
      self.funcIdx[self.charIdx['h']] = self.renderHelp
      self.funcIdx[self.charIdx['v']] = self.renderDefaults
      self.funcIdx[self.charIdx['c']] = self.renderCounts
      self.funcIdx[self.charIdx['f']] = self.renderFood
      self.funcIdx[self.charIdx['w']] = self.renderWater
      self.funcIdx[self.charIdx['q']] = self.renderHalf
      self.funcIdx[self.charIdx['d']] = self.renderDeps
      self.funcIdx[self.charIdx['a']] = self.renderAction
      self.funcIdx[self.charIdx['1']] = self.renderMelee
      self.funcIdx[self.charIdx['2']] = self.renderRange
      self.funcIdx[self.charIdx['3']] = self.renderMage
      self.funcIdx[self.charIdx[']']] = self.renderRight

      self.renderHelp(True)

   def gridApply(self, damageF):
      rets = []
      for r in range(self.conf.R):
         for c in range(self.conf.C):
             dist = Attack.l1((r, c), self.conf.SPAWN)
             rets.append(((r,c), damageF(dist)))
      return rets

   def renderMelee(self, update=False):
      damageF = self.conf.MELEEDAMAGE
      vals = self.gridApply(damageF)
      self.renderVals(env, vals, update=update)

   def renderRange(self, update=False):
      damageF = self.conf.RANGEDAMAGE
      vals = self.gridApply(damageF)
      self.renderVals(env, vals, update=update)

   def renderMage(self, update=False):
      damageF = self.conf.MAGEDAMAGE
      vals = self.gridApply(damageF)
      self.renderVals(env, vals, update=update)

   def renderRight(self, update=False):
      print("Saved right screenshot")
      surf = self.dat
      im = pygame.surfarray.array3d(surf)
      im = im.transpose((1, 0, 2))
      imsave("dat{}.png".format(self.screenshot_idx), im)

      self.screenshot_idx += 1

   def renderHelp(self, update=False):
      if update:
         self.dat = self.help.render()

   def renderAction(self, update=False):
      #if update:
      self.dat = self.action.render()
 
   def renderFood(self, update=False):
      if update:
         self.val = valF(0, 36)
      self.renderVals(env, self.val, update)

   def renderWater(self, update=False):
      if update:
         self.val = valF(36, 0)
      self.renderVals(env, self.val, update)

   def renderHalf(self, update=False):
      if update:
         self.val = valF(18, 18)
      self.renderVals(env, self.val, update)

   def renderEmpty(self, update=False):
      if update:
         self.val = valF(0, 0)
      self.renderVals(env, self.val, update)

   def renderDefaults(self, update=False):
      self.renderVals(update)

   def renderVals(self, update=False):
      self.dat = self.vals.render(self.trans, update)
      self.dat.set_alpha(255)
      if self.transparent:
         self.dat.set_alpha(0)
         env = self.env.copy()
         env.blit(self.dat, (0, 0), 
               special_flags=pygame.BLEND_RGB_ADD)
         #env.blit(self.dat, (0, 0))
         self.dat = env
         
   def renderDeps(self, update=False):
      if update:
         self.dat = self.deps.render()

   def renderCounts(self, update=False):
      self.dat = self.counts.render(self.trans, update)

   def render(self, env, trans, update=False):
      #self.ents, self.valF, self.trans = ents, valF, trans
      self.env = env
      self.reset()
      self.trans = trans
      self.funcIdx[self.idx](update)
      self.surf.blit(self.dat, (0, 0))
      self.renderBorder()
      return self.surf

   def key(self, c):
      old_idx = self.idx
      if c == 'o':
         self.toggle()
      elif c == 't':
         self.transparent = not self.transparent
      elif c in self.charIdx:
         self.idx = self.charIdx[c]
      else:
         return

      if old_idx == self.charIdx[']'] and self.idx == old_idx:
         return

      self.funcIdx[self.idx](update=True)

      if c == ']':
         self.idx = old_idx

   def toggle(self):
      self.idx = (self.idx + 1) % self.nIdx

class Title(embyr.Container):
   def __init__(self, fonts, canvasSize, **kwargs):
      super().__init__(canvasSize, **kwargs)
      godsword = embyr.Texture('resource/Splash/ags.png', mask=Neon.MASK.rgb)
      w, h = godsword.get_width(), godsword.get_height()
      scaled = int(w*self.H/h), int(self.H)
      self.godsword = pygame.transform.scale(godsword, scaled)

   def render(self):
      self.reset()
      self.blit(self.godsword, (0, 0))
      return self.surf

class FPS(embyr.Container):
   def __init__(self, fonts, canvasSize, **kwargs):
      super().__init__(canvasSize, **kwargs)
      self.fpsTracker = embyr.FPSTracker()
      self.fonts = fonts

   def render(self):
      self.reset()
      self.fpsTracker.update()
      fps = self.fonts.Large.render('FPS: '+self.fpsTracker.fps, 1, Neon.GOLD.rgb)
      self.surf.blit(fps, (50,10))
      return self.surf

class Corner(embyr.Container):
   def __init__(self, env, fonts, canvasSize, **kwargs):
      super().__init__(canvasSize, **kwargs)
      self.title = Title(fonts, (256, 64), border=3) 
      self.fps = FPS(fonts, (256, 64), border=3)

   def render(self):
      self.reset()
      title = self.title.render()
      fps   = self.fps.render()

      self.surf.blit(title, (0, 0))
      self.surf.blit(fps, (0, 64))
      return self.surf

#Work on
class LeftSidebar(embyr.Container):
    def __init__(self, realm, fonts, canvasSize, **kwargs):
      super().__init__(canvasSize, **kwargs)
      self.realm, self.fonts = realm, fonts

    def render(self):
      self.reset()
      stats = self.realm.world.stats
      render.renderExchange(self, stats, self.fonts, self.W, self.H)
      return self.surf
  
class Histogram(embyr.Container):
   def __init__(self, fonts, canvasSize, **kwargs):
      super().__init__(canvasSize, **kwargs)
      self.fonts = fonts

   def render(self, dat):
      self.reset()
      render.renderHist(self, dat, self.fonts, self.W, self.H)
      return self.surf

class Graph(embyr.Container):
   def __init__(self, fonts, canvasSize, **kwargs):
      super().__init__(canvasSize, **kwargs)
      self.fonts = fonts

   def render(self, dat):
      self.reset()
      render.renderGraph(self, dat, self.fonts, self.W, self.H, color=Neon.GOLD.rgb)
      return self.surf

class BottomSidebar(embyr.Container):
   def __init__(self, realm, fonts, canvasSize, **kwargs):
      super().__init__(canvasSize, **kwargs)
      self.hist  = Histogram(fonts, (self.W, self.H//2), border=3)
      self.graph = Graph(fonts, (self.W, self.H//2), border=3)
      self.realm = realm

   def render(self):
      self.fill(Neon.BLACK.rgb)
      stats = self.realm.world.stats

      #Render histograms of time alive, levels, and best agents
      lifespans = [[e.timeAlive] for e in stats.pcs]
      #timesAlive = Render.histogram(timesAlive)
      #combatStats = [(e.melee, e.ranged, e.defense) for e in blocks]

      hist  = self.hist.render(lifespans)
      graph = self.graph.render(stats.numEntities[-self.W:])

      self.surf.blit(hist,  (0, 0))
      self.surf.blit(graph, (0, self.H//2))

      #Render
      #Render.renderHist(self.canvas, self.fonts, lifespans, self.W, self.H, 0, self.H//2)
      #renderHist(screen, fonts, combatStats, 0, H+h, W, h)
      #Render.renderGraph(self.canvas, stats.numEntities[-self.W:], self.fonts, self.W, self.H, color=Color.GOLD)

      # Render.renderGraphs(self.canvas, stats, self.fonts, 
      #      self.W, self.H)
      return self.surf
 
class Counts(embyr.Container):
   def __init__(self, realm, mipLevels, canvasSize, **kwargs):
      super().__init__(canvasSize, **kwargs)
      self.R, self.C = realm.world.env.tiles.shape
      self.nColors = realm.world.env.nCounts
      self.mipLevels = mipLevels
      self.realm = realm
      self.tileSz = 16

   def render(self, trans, update):
      env = self.realm.world.env
      tiles = env.tiles
      R, C = tiles.shape
      counts = np.array([tile.counts for tile in tiles.ravel()])
      nCounts = counts.shape[1]
      counts = counts.reshape(R, C, counts.shape[1])
      counts = [counts[:, :, i] for i in range(counts.shape[2])]
      if nCounts <= 12:
         colors = Neon.color12()
      else:
         colors = Color256.colors[0::256//nCounts]
      for idx, count in enumerate(counts):
         counts[idx] = 20*counts[idx][:, :, np.newaxis]*np.array(colors[idx].value)/255.0
      sumCounts = sum(counts)

      R, C, _ = sumCounts.shape
      counts = np.clip(sumCounts, 0, 255).astype(np.uint8)
      counts = np.clip(counts, 0, 255).astype(np.uint8)

      valCrop, txSz = embyr.mapCrop(self, counts,
            self.tileSz, trans)
      txSz = embyr.mipKey(txSz, self.mipLevels)
      embyr.renderMap(self, valCrop, txSz)
      return self.surf

class Deps(embyr.Container):
   def __init__(self, realm, canvasSize, **kwargs):
      super().__init__(canvasSize, **kwargs)
      self.R, self.C = realm.world.env.tiles.shape
      self.realm = realm
      self.tileSz = 16

   def render(self):
      anns = self.realm.sword.anns
      deps = [ann.visDeps() for ann in anns]
      grids = []
      for dep in deps:
         vals = dep
         vList = [e[1] for e in vals]
         vMean, vStd = np.mean(vList), np.std(vList)+1e-3
         nStd, nTol = 4.0, 0.5
         grayVal = int(255 / nStd * nTol)
         grid = np.zeros((16, 16, 3))
         for v in vals:
            pos, mat = v
            r, c = pos
            mat = (mat - vMean) / vStd
            color = np.clip(mat, -nStd, nStd)
            color = int(color * 255.0 / nStd)
            if color > 0:
                color = (0, color, 128)
            else:
                color = (-color, 0, 128)
            grid[r, c] = color
         grids.append(grid)
      grids += 15*[0*grids[0]]
      grids = grids[:16]
      grids1 = np.vstack(grids[:4])
      grids2 = np.vstack(grids[4:8])
      grids3 = np.vstack(grids[8:12])
      grids4 = np.vstack(grids[12:16])
      grids  = np.hstack((grids1, grids2, grids3, grids4))
      embyr.renderMap(self, grids, self.tileSz)
      return self.surf

class Action(embyr.Container):
   def __init__(self, realm, 
            canvasSize, **kwargs):
      super().__init__(canvasSize, **kwargs)
      self.realm = realm
      self.makeMaps()
      self.tileSz = 16
      self.RED   = np.array(Defaults.RED)
      self.GREEN = np.array(Defaults.GREEN)
      self.BLUE  = np.array(Defaults.BLUE)

   def render(self):
      ents = [e.server for e in self.realm.desciples.values()]
      self.updateMaps(ents)
      maps = self.maps
      maps = maps + 15*[0*maps[0]]
      maps = maps[:16]
      mapv1 = np.vstack(maps[:4])
      mapv2 = np.vstack(maps[4:8])
      mapv3 = np.vstack(maps[8:12])
      mapv4 = np.vstack(maps[12:16])
      maps = np.hstack((mapv1, mapv2, mapv3, mapv4))

      maps = maps.astype(np.uint8)
      #maps = pygame.pixelcopy.make_surface(maps)
      #valCrop, txSz = embyr.mapCrop(self, maps,
      #      self.tileSz, trans)
      #txSz = embyr.mipKey(txSz, self.mipLevels)
      embyr.renderMap(self, maps, int(1.7*self.tileSz))
      return self.surf

   def updateMaps(self, ents):
      for ent in ents:
         attack = ent.attack.action
         #targ   = ent.attack.args
         targ = ent.targ
         if targ is None or targ.damage is None:
            continue
         idx    = ent.colorInd
         if idx >= 16:
            continue
         entr, entc = ent.attkPos
         targr, targc = ent.targPos
         r = self.MAPCENT + targr - entr
         c = self.MAPCENT + targc - entc

         if issubclass(attack, action.Melee):
            self.counts[idx][r, c, 0] += 1
         elif issubclass(attack, action.Range):
            self.counts[idx][r, c, 1] += 1
         elif issubclass(attack, action.Mage):
            self.counts[idx][r, c, 2] += 1

         rgb = self.counts[idx][r, c]
         normSum = np.sum(rgb)
         redVal   = self.RED   * rgb[0] / normSum 
         greenVal = self.GREEN * rgb[1] / normSum 
         blueVal  = self.BLUE  * rgb[2] / normSum 
         color = (redVal + greenVal + blueVal).astype(np.uint8)
         self.maps[idx][r, c] = color
         #colorIdx = np.argmax(self.counts[idx][r, c])
         #self.maps[idx][r, c] = self.colors[colorIdx]
 
   def makeMaps(self):
      self.NMAPS = 16#self.config.NPOP
      self.MAPCENT = 4 #self.config.STIM
      self.MAPSZ = 2*self.MAPCENT + 1

      self.maps = []
      self.counts = []
      for idx in range(self.NMAPS):
         self.maps.append(np.zeros((self.MAPSZ, self.MAPSZ, 3)))
         self.counts.append(np.zeros((self.MAPSZ, self.MAPSZ, 3)))


   def renderEnts(self, trans):
      self.reset()
      R, C = self.env.shape
      offs, txSz = embyr.offsets(self, R, C, self.tileSz, trans)
      txSz = embyr.mipKey(txSz, self.mipLevels)
      minH, _, minW, _ = offs
      self.sprites.render(self, (minH, minW), txSz, self.iso)
      return self.surf
 

   def refresh(self, trans, iso):
      self.iso = iso
      return self.render(None, trans, keyframe=False)

class Vals(embyr.Container):
   def __init__(self, realm, mipLevels, 
            canvasSize, conf, **kwargs):
      super().__init__(canvasSize, **kwargs)
      self.R, self.C = realm.world.env.tiles.shape
      self.vals = np.zeros((self.R, self.C), dtype=object)
      self.mipLevels = mipLevels
      self.realm = realm
      self.tileSz = 16

   def update(self):
      ann = self.realm.sword.anns[0]
      vals = ann.visVals()
      vList = [e[1] for e in vals]
      vMean, vStd = np.mean(vList), np.std(vList)
      nStd, nTol = 4.0, 0.5
      grayVal = int(255 / nStd * nTol)
      for v in vals:
         pos, mat = v
         r, c = pos
         mat = (mat - vMean) / vStd
         color = np.clip(mat, -nStd, nStd)
         color = int(color * 255.0 / nStd)
         if color > 0:
             color = (0, color, 128)
         else:
             color = (-color, 0, 128)
         self.vals[r, c] = color

   def render(self, trans, update):
      self.reset()
      if update:
         self.update()

      valCrop, txSz = embyr.mapCrop(self, self.vals, 
            self.tileSz, trans)
      txSz = embyr.mipKey(txSz, self.mipLevels)
      embyr.renderMap(self, valCrop, txSz)
      return self.surf

class Help(embyr.Container):
   def __init__(self, fonts, canvasSize, **kwargs):
      super().__init__(canvasSize, **kwargs)
      self.fonts = fonts
      self.text = [
              'Command List:',
              '   Environment',
              '      i: toggle isometric',
              '   Overlay:',
              '      o: cycle overlay',
              '      h: help overlay',
              '      c: count overlay',
              '      v: value overlay',
              '      t: toggle transparency',
              '      w: value no water',
              '      f: value no food',
              '      q: value half food/water',
              '      a: value no food/water',
              '      r: screenshot the right side of screen',
              ]

   def render(self):
      self.reset()
      offset, margin, pos = 64, 16, 0
      for item in self.text:
         text = self.fonts.Huge.render(item, 1, Neon.GOLD.rgb)
         self.blit(text, (margin, pos+margin))
         pos += offset
      return self.surf
 
class Canvas(embyr.Container):
    def __init__(self, size, realm, dims, conf, **kwargs):
        super().__init__(size, **kwargs)
        self.realm, border, self.iso = realm, 3, False
        self.W, self.H, self.side = dims
        mipLevels = [8, 16, 32, 48, 64, 128]
        textures = TextureInitializer(16, mipLevels = mipLevels)
        self.fonts = renderutils.Fonts('resource/Fonts/dragonslapper.ttf')

        self.envViewport   = EnvViewport(realm, textures, 
              mipLevels, self.fonts, (self.H, self.H), self.iso)

        self.datViewport   = DatViewport(realm, textures, 
              self.fonts, (self.H, self.H), conf, border=border)
        self.leftSidebar   = LeftSidebar(realm, self.fonts, 
              (self.side, self.H), border=border)
        self.bottomSidebar = BottomSidebar(realm, self.fonts, (self.W, self.side))
        self.corner        = Corner(realm, self.fonts, (self.side, self.side))

    def render(self, realm, trans, overlay=True):
        #env, stats, valF
        #stats.pcs
        env = self.envViewport.render(trans, overlay)
        dat = self.datViewport.render(env, trans, False)
        corner        = self.corner.render()
        self.env = env
        
        self.surf.blit(env, (self.side, 0))
        self.surf.blit(dat, (self.H+self.side, 0))
        self.surf.blit(corner, (0, self.H))

        if overlay:
            leftSidebar   = self.leftSidebar.render()
            bottomSidebar = self.bottomSidebar.render()

            self.surf.blit(leftSidebar, (0, 0))
            self.surf.blit(bottomSidebar, (self.side, self.H))

        return self.surf

    def renderTitle(self):
        godsword = renderutils.pgRead('resource/Splash/agsfull.png', mask=Neon.MASK.rgb)
        w, h = godsword.get_width(), godsword.get_height()
        W, H = self.size
        ratio = W/w
        w, h = int(ratio*w), int(ratio*h)
        godsword = pygame.transform.scale(godsword, (w, h))
        hOff = int(H/2 - godsword.get_height()/2)
        self.blit(godsword, (0, hOff))
        return self.surf

    def toggleEnv(self, trans):
        sx, sy, tx, ty = trans
        self.iso = not self.iso
        env = self.envViewport.refresh(trans, self.iso)
        self.surf.set_clip((self.side, 0, self.H, self.H))
        env = pygame.transform.scale(env, (sx, sy))
        self.surf.blit(env, (self.side+tx, ty))
        self.surf.set_clip(None)

    def key(self, c):
        self.datViewport.key(c)

    def leftScreenshot(self):
        self.envViewport.leftScreenshot()
