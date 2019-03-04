from pdb import set_trace as T

from scipy.misc import imread
from scipy.misc import imsave
from sim.lib import Enums
from sim.lib.Enums import Material
import time
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
   def __init__(self, sz, root='resource/', scale=1):
      self.width       = sz
      self.statHeight  = 2
      self.scale = scale

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
         mat.tex = pygame.transform.scale(mat.tex, (int(self.width*self.scale), 
                int(self.width*self.scale)))
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
   def __init__(self, w, h, load, res=4):
      self.W, self.H = 2*512, 2*512
      self.deltaX, self.deltaY = 0, 0
      self.fDir = 'resource/map/'
      self.fName = 'profs'
      self.env = np.ones((h, w), dtype=int)
      self.textures = TextureInitializer(16 )
      self.client = Client(res)
      self.resInd = res
      self.setupScreen(self.W, self.H)
      self.loadIf(load)
      self.start = time.time()

   def save(self): 
      np.savetxt(self.fDir + self.fName + '.txt', self.env, fmt='%d')
      pygame.image.save(self.screen, self.fDir + self.fName + '.png')

   def loadIf(self, load):
      if load:
         self.env = np.genfromtxt(self.fDir + self.fName + '.txt')
         #self.env = imread(self.fDir + self.fName + '.png')
      self.redraw()

   def redraw(self):
      self.screen.fill((0, 0, 0))
      R, C = int(self.H//2**self.resInd), int(self.W//2**self.resInd)
      RR, CC = self.env.shape
      for r in range(R):
         for c in range(C):
            rr, cc = r+self.deltaY, c+self.deltaX
            if rr < 0 or cc < 0 or rr >= RR or cc >= CC:
               continue
            tex = self.env[rr, cc]
            self.renderTile(r, c, self.textures.material[tex])

   def setupScreen(self, W, H):
      pygame.init()
      self.screen    = pygame.display.set_mode((W, H)) 

   def getTile(self, rPx, cPx):
      return int(rPx//2**self.resInd), int(cPx//2**self.resInd)

   def renderTile(self, r, c, tex):
      w = 2**self.resInd
      tex = pygame.transform.scale(tex, (w, w))
      self.screen.blit(tex, (c*w, r*w))

   def render(self):
      #Draw
      if self.client.clicked:
         x, y = pygame.mouse.get_pos()
         c, r = self.getTile(x, y)
         rr, cc = r+self.deltaY, c+self.deltaX
         tex = self.textures.material[self.client.tex]
         R, C = self.env.shape
         if rr >= 0 and cc >= 0 and rr<R and cc<C:
            self.env[rr, cc] = str(self.client.tex)
            self.renderTile(r, c, tex)

      #buff = pygame.transform.scale(self.buffer, (sx, sy))

      #Render
      #self.screen.blit(buff, (tx, ty))
      pygame.display.flip()

   def update(self):
      self.client.update()
      if self.client.resInd != self.resInd:
         self.resInd = self.client.resInd
         self.redraw()
      if self.client.deltaX != self.deltaX:
         self.deltaX = self.client.deltaX 
         self.redraw()
      if self.client.deltaY != self.deltaY:
         self.deltaY = self.client.deltaY 
         self.redraw()

      self.render()
      if time.time() - self.start > 5.0:
         self.start = time.time()
         self.save()

class Client:
   def __init__(self, resInd):
      self.volX, self.volY = 0, 0
      self.volDeltaX, self.volDeltaY = 0, 0
      self.deltaX, self.deltaY = 0, 0
      self.clicked = False
      self.rightClicked = False
      self.tex = 1
      self.resInd = resInd
      self.delta = 10

   def update(self):
      self.processEvents(pygame.event.get())

   def quit(self):
      pygame.quit()
      sys.exit()

   def mouseDown(self, button):
      if button == 1 and not self.clicked:
         self.volX, self.volY = pygame.mouse.get_pos()
         self.clicked = True
      if button == 3 and not self.rightClicked:
         self.volX, self.volY = pygame.mouse.get_pos()
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
         if self.resInd < 6:
            self.resInd += 1
      elif button == 5:
         if self.resInd > 3:
            self.resInd -= 1
      print(self.resInd)

   def keyUp(self, button):
      if button == pygame.K_a:
         self.deltaX -= self.delta
      elif button == pygame.K_d:
         self.deltaX += self.delta
      elif button == pygame.K_s:
         self.deltaY += self.delta
      elif button == pygame.K_w:
         self.deltaY -= self.delta
      elif button == pygame.K_e:
         if self.tex < 6:
            self.tex += 1
      elif button == pygame.K_q:
         if self.tex > 0:
            self.tex -= 1

   def processEvents(self, events):
      for e in events:
         if e.type == pygame.QUIT:
            self.quit()
         elif e.type == pygame.MOUSEBUTTONDOWN:
            self.mouseDown(e.button)
         elif e.type == pygame.MOUSEBUTTONUP:
            self.mouseUp(e.button)
         elif e.type == pygame.KEYUP:
            print('KeyUp')
            self.keyUp(e.key)

if __name__ == '__main__':
   w, h, load = int(sys.argv[1]), int(sys.argv[2]), True
   if len(sys.argv) > 3 and sys.argv[3] == '--noload':
       load=False
   mapMaker = MapMaker(w, h, load)
   while True:
      mapMaker.update()
