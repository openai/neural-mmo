#Main renderer. Loads all textures, takes render information from
#the world, applies texturing for env/entities/actions/additional
#statistic visuals

from pdb import set_trace as T
import pygame
import sys
import time

import numpy as np
from scipy.misc import imread
from enum import Enum

from forge.embyr import embyr
from forge.embyr import utils as renderutils
from forge.blade.action import action
from forge.blade.action import v2
from forge.blade.lib.enums import Neon
        
def renderEnts(screen, ent, fonts, entTx, atnTx, txSz, iso):
   H, W = ent.shape
   for h in range(H):
      for w in range(W):
         if ent[h, w] is None:
             continue
         for e in ent[h, w]:
            renderSprite(screen, e, (h, w), fonts, entTx, 
                  atnTx, txSz, iso)

def renderVals(screen, vals, mipLevels, txSz):
   H, W = vals.shape
   for h in range(H):
      for w in range(W):
         if vals[h, w] is 0:
             continue
         v = vals[h, w]
         mask = np.array(mipLevels) >= txSz
         key = np.array(mipLevels)[np.argmax(mask)]
         ww, hh = cartCoords(w, h, key)
         screen.rect(v, (ww, hh, key, key))

def renderSprite(screen, sprite, offs, tex, fonts, txSz, iso):
   renderEntity(screen, sprite, offs, tex, fonts, txSz, iso)
   renderStats(screen,  sprite, offs, tex, fonts, txSz, iso)
   if sprite.targ is not None:
      renderAction(screen, sprite, offs, tex, fonts, txSz, iso)

def renderEntity(screen, sprite, offs, tex, fonts, txSz, iso):
   rOff, cOff = offs
   r, c = sprite.posAnim
   W, H = screen.size
   pos = embyr.tileCoords(c-cOff, r-rOff, W, H, txSz, iso)
   screen.blit(tex.entity[txSz][sprite.ent.color.name], pos)

def renderAction(screen, sprite, offs, tex, fonts, txSz, iso):
   rAnim, cAnim = sprite.posAnim
   rTarg, cTarg = sprite.targ.pos
   rTrans, cTrans= offs
   frame = sprite.frame

   rOff, cOff = rTarg - rAnim, cTarg - cAnim
   r = rAnim + frame * rOff 
   c = cAnim + frame * cOff 

   e = sprite.ent
   W, H = screen.W, screen.H
   if e.entID != sprite.targ.entID:
      attk = e.attack.action
      w, h = embyr.tileCoords(c-cTrans, r-rTrans, W, H, txSz, iso)
      if attk is v2.Melee:
         angle = np.arctan2(rAnim-rTarg, cTarg-cAnim) * 180 / np.pi - 90
         tex = tex.action[txSz]['melee']
         tex = pygame.transform.rotate(tex, angle + 360*frame)
      elif attk is v2.Range:
         angle = np.arctan2(rAnim-rTarg, cTarg-cAnim) * 180 / np.pi - 90
         tex = tex.action[txSz]['range']
         tex = pygame.transform.rotate(tex, angle)
      elif attk is v2.Mage:
         tex = tex.action[txSz]['mage']
         sz = int(txSz * frame)
         tex = pygame.transform.scale(tex, (sz, sz))
      else:
         print('invalid attack texture in render.py')
         exit(0)

      screen.blit(tex, (w, h))
      #w, h = embyr.tileCoords(cAnim-cOff, rAnim-rOff, W, H, txSz, iso)
      #screen.rect(Neon.BLUE.rgb, (w, h, txSz, txSz), 2)

      #w, h = embyr.tileCoords(cTarg-cOff, rTarg-rOff, W, H, txSz, iso)
      #screen.rect(Neon.ORANGE.rgb, (w, h, txSz, txSz), 4)

   damage = e.damage
   if damage is not None:
      w, h = embyr.tileCoords(cAnim-cTrans, rAnim-rTrans, W, H, txSz, iso)
      if damage == 0:
         text = fonts.huge.render(str(damage), 1, Neon.BLUE.rgb)
      else:
         text = fonts.huge.render(str(damage), 1, Neon.RED.rgb)
      screen.blit(text, (w, h-16))

def renderStat(screen, stat, maxStat, statInd, w, h, txSz, color):
   #w, h = tileCoords(w, h, txSz)
   barSz = (1.5*txSz) // 16
   scale = txSz / maxStat
   scaledStat = int(scale*stat)
   if stat > 0:
      screen.rect(color, (w, h+statInd*barSz, scaledStat, barSz))
   if maxStat-stat > 0:
      screen.rect(Neon.RED.rgb,
            (w+scaledStat, h+statInd*barSz, txSz-scaledStat, barSz))

def renderStats(screen, sprite, offs, tex, fonts, txSz, iso):
   e = sprite.ent
   rOff, cOff = offs
   r, c = sprite.posAnim
   W, H = screen.W, screen.H
   w, h = embyr.tileCoords(c-cOff, r-rOff, W, H, txSz, iso)
   stats = []
   stats.append((e.health, e.maxHealth, Neon.GREEN.rgb))
   stats.append((e.water, e.maxWater, Neon.BLUE.rgb))
   stats.append((e.food, e.maxFood, Neon.GOLD.rgb))

   text = fonts.large.render(sprite.entName, 1, sprite.nameColor.rgb)
   screen.blit(text, (w-40, h-20))

   for (ind, dat) in enumerate(stats):
      s, sMax, color = dat
      renderStat(screen, s, sMax, ind, w, h, txSz, color)

def renderGraph(screen, dat, fonts, w, h, border=2, 
      bg=Neon.BLACK.rgb, color=Neon.RED.rgb):
   if len(dat) == 0:
      return

   tickHeight = 5
   fontWidth = 40
   fontHeight = 40
   yy = dat[-w:]
   xx = np.arange(len(yy))
   for x, y in zip(xx, yy):
      screen.line(color, (x, h-y), (x, h-(y+tickHeight)))

   #screen.rect(Color.RED, 
   #      (x - fontWidth+3, h-(y+tickHeight+3), fontWidth, tickHeight))

   text = fonts.large.render(str(y), 1, Neon.YELLOW.rgb)
   screen.blit(text, (x-fontWidth, h-y-fontHeight))

def renderHist(screen, dat, fonts, W, H, mul=1, border=4, 
      bg=Neon.BLACK.rgb, colors=(Neon.GREEN.rgb, Neon.RED.rgb, Neon.BLUE.rgb)):
   px = 8
   spacer = 3
   blockSize = 64
   valSize = 32
   txtW = 45
   txtH = 20
   leftPad = 16
   barSz = 8
   delta = 0
   colorInd = 0
   x = 0 
   scale = 1
   if np.max(dat) > H:
       scale = H / np.max(dat)
   for block in dat:
      for origVal in block: 
         val = int(origVal / scale)
         #val = int(100*np.random.rand())
         color = colors[colorInd % len(block)] 
         xx, yy = x+border+barSz, H-val-border
         screen.rect(color, (xx, yy, barSz, val))

         x += valSize
         #if ww > graphW:
         #   return
         text = fonts.small.render(str(origVal), 1, Neon.YELLOW.rgb)
         screen.blit(text, (xx-txtW, yy-txtH))

         #delta += px+spacer
         #colorInd += 1

      #delta += setSpacer

def histogram(data, numBins=10):
   data = sorted(data, reverse=True)
   split = int(np.ceil(len(data)/float(numBins)))
   hist = []
   for i in range(numBins):
      val = 0
      datSplit = data[split*i:split*(i+1)]
      if len(datSplit) > 0:
         val = int(np.mean(datSplit))
      hist += [[val]]
   return hist

def renderGraphs(screen, stats, fonts, W, H, border=4):
   #Render histograms of time alive, levels, and best agents

   blocks = stats.statBlocks
   timesAlive = np.asarray([[e.timeAlive] for e in blocks])
   timesAlive = histogram(timesAlive)
   combatStats = [(e.melee, e.ranged, e.defense) for e in blocks]
   
   #Render
   renderHist(screen, fonts, timesAlive, 0, H, w, h)
   #renderHist(screen, fonts, combatStats, 0, H+h, W, h)
   renderGraph(screen, stats.numEntities[-W:], fonts, W, H,
         color=Neon.GOLD.rgb)
 
def renderExchangeBlock(screen, entry, fonts, ind, 
      W, H, blockH, pad=4):
   screen.rect(Neon.RED.rgb, 
         (0, (ind+1)*blockH, W, 3), 0)

   numBuy, numSell = entry.numBuy, entry.numSell
   maxBuyPrice, minSellPrice = entry.maxBuyPrice, entry.maxSellPrice
   color = Neon.YELLOW.rgb

   text = []
   text.append(entry.itemName)
   text.append('Buy/Sell Offers:  ' + str(numBuy) + ',  ' + str(numSell))
   text.append('Min/Max Price:  ' + str(maxBuyPrice) + ',  ' + str(minSellPrice))

   def height(i, inc):
      return i*blockH+ int(inc*blockH/5.0)

   for i, e in enumerate(text):
      txt = fonts.normal.render(e, 1, color)
      screen.blit(txt, (pad, height(ind, i)))
  
def renderExchange(screen, stats, fonts, W, H):
   blockH = W/2
   numRender = H / blockH
   exchange = stats.exchange 

   i = -1
   for e in exchange.queue:
      i += 1
      if i == numRender:
         break

      renderExchangeBlock(screen, e, fonts, i, W, H, blockH)
