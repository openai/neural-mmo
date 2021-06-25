from pdb import set_trace as T

from scipy.misc import imread
from scipy.misc import imsave
from sim.lib import Enums
from sim.lib.Enums import Material
import sys
import numpy as np
import pygame

def readRGB(path):
   return imread(path)[:, :, :3]

def pgRead(path, alpha=False, rot=90):
   try:
      img = readRGB(path)
   except FileNotFoundError:
      return None
   img = pygame.pixelcopy.make_surface(img)
   if alpha:
      img.set_colorkey((255, 255, 255))
   return pygame.transform.rotate(img, rot)

class TextureInitializer():
   def __init__(self, sz, root='resource/'):
      self.width       = sz
      self.statHeight  = 2

      self.material  = readRGB(root+'Material/textures.png')
      self.material  = self.textureTiles()
      self.entity    = self.textureFromFile(Enums.Entity, root+'Entity/')

   def textureTiles(self):
      reverse = {}
      for mat in Material:
         mat = mat.value
         texCoords = mat.tex
         tex = self.getTile(*texCoords)
         tex = pygame.pixelcopy.make_surface(tex)
         mat.tex = pygame.transform.rotate(tex, 90)
         reverse[mat.index] = mat.tex
      return reverse

   def getTile(self, r, c):
      w = self.width
      tile = self.material[r*w:r*w+w, c*w:c*w+w, :]
      return tile

   def textureFromFile(self, enum, path, alpha=True, rot=270):
      reverse = {}
      for e in enum:
         texPath = path + e.name.lower() + '.png'
         tex = pgRead(texPath, alpha=alpha, rot=rot)
         e.value.tex = tex
         if type(e.value.data) == tuple:
            reverse[e.value.data[0]] = tex
         else:
            reverse[e.value.data] = tex

      return reverse

class MapMaker:
   def __init__(self, w, h, res=16):
      self.W, self.H, self.res = w, h, res
      self.env = np.zeros((h, w), dtype=np.uint8)
      self.textures = TextureInitializer(self.res)

      self.setupScreen(self.W, self.H)

      self.zoom = 1
      self.maxZoom = int(np.max(list(self.textures.material.keys())))
      self.zoomDelta = 1
      self.deltaX, self.deltaY = 0, 0
      self.volDeltaX, self.volDeltaY = 0, 0
      self.clicked = False
      self.rightClicked = False

   
   def getTile(self, rPx, cPx):
      return rPx//self.res, cPx//self.res

   def renderTile(self, r, c, tex):
      w = self.res
      self.buffer.blit(tex, (c*w, r*w))

   def render(self):
      self.screen.fill((0, 0, 0))

      #Draw
      if self.rightClicked:
         x, y = pygame.mouse.get_pos()
         x = x - self.deltaX
         y = y - self.deltaY
         c, r = self.getTile(x, y)
         tex = self.textures.material[self.zoom]
         self.env[r, c] = np.uint8(self.zoom)
         self.renderTile(r, c, tex)

      #Scale
      scaleX, scaleY = int(self.H), int(self.W)
      buff = pygame.transform.scale(self.buffer, (scaleX, scaleY))

      #Translate
      deltaX = self.deltaX + self.volDeltaX - scaleX//2 + self.W//2
      deltaY = self.deltaY + self.volDeltaY - scaleY//2 + self.H//2

      #Render
      self.screen.blit(buff, (deltaX, deltaY))
      pygame.display.flip()

   def setupScreen(self, envR, envC):
      self.W = envC * self.res
      self.H = envR * self.res

      pygame.init()
      self.screen    = pygame.display.set_mode((self.W, self.H)) 
      self.buffer    = pygame.surface.Surface((self.W, self.H))

   def update(self):
      self.processEvents(pygame.event.get())
      self.updateMouse()
      imsave('resource/map/smallmap.png', self.env)
      self.render()

   def updateMouse(self):
      if self.clicked:
         volX, volY = self.volX, self.volY
         curX, curY = pygame.mouse.get_pos()
         self.volDeltaX = curX - volX
         self.volDeltaY = curY - volY

   def quit(self):
      pygame.quit()
      sys.exit()

   def mouseDown(self, button):
      if button == 1 and not self.clicked:
         self.volX, self.volY = pygame.mouse.get_pos()
         self.clicked = True
      if button == 3:
         self.rightClicked = True

   def mouseUp(self, button):
      if button == 1:
         if self.clicked:
            self.deltaX += self.volDeltaX
            self.deltaY += self.volDeltaY
            self.volDeltaX, self.volDeltaY = 0, 0
            self.clicked = False
      elif button == 3:
         self.rightClicked = False
      elif button == 4:
         if self.zoom < self.maxZoom:
            self.zoom += self.zoomDelta
      elif button == 5:
         if self.zoom > 0:
            self.zoom -= self.zoomDelta

   def processEvents(self, events):
      for e in events:
         if e.type == pygame.QUIT:
            self.quit()
         elif e.type == pygame.MOUSEBUTTONDOWN:
            self.mouseDown(e.button)
         elif e.type == pygame.MOUSEBUTTONUP:
            self.mouseUp(e.button)



if __name__ == '__main__':
   w, h = int(sys.argv[1]), int(sys.argv[2])
   mapMaker = MapMaker(w, h)
   while True:
      mapMaker.update()
