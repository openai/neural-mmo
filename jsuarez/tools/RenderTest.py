from pdb import set_trace as T
import pygame
from pygame.surface import Surface
import numpy as np
import time
from enum import Enum

class Neon(Enum):
   def rgb(h):
      h = h.lstrip('#')
      return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

   RED      = rgb('#ff0000')
   ORANGE   = rgb('#ff8000')
   YELLOW   = rgb('#ffff00')

   GREEN    = rgb('#00ff00')
   MINT     = rgb('#00ff80')
   CYAN     = rgb('#00ffff')

   BLUE     = rgb('#0000ff')
   PURPLE   = rgb('#8000ff')
   MAGENTA  = rgb('#ff00ff')

   WHITE    = rgb('#ffffff')
   GRAY     = rgb('#666666')
   BLACK    = rgb('#000000')

   BLOOD    = rgb('#bb0000')
   BROWN    = rgb('#7a3402')
   GOLD     = rgb('#eec600') #238 198
   SILVER   = rgb('#b8b8b8')

   FUCHSIA  = rgb('#ff0080')
   SPRING   = rgb('#80ff80')
   SKY      = rgb('#0080ff')
   TERM     = rgb('#41ff00')

   def rand12():
      ind = np.random.randint(0, 12)
      return (
              Neon.RED, Neon.ORANGE, Neon.YELLOW,
              Neon.GREEN, Neon.MINT, Neon.CYAN,
              Neon.BLUE, Neon.PURPLE, Neon.MAGENTA,
              Neon.FUCHSIA, Neon.SPRING, Neon.SKY)[ind].value

class Container:
   def __init__(self, size, border=0, reset=False, key=None):
      self.W, self.H = size
      self.canvas = Surface((self.W, self.H))
      if key is not None:
         self.canvas.set_colorkey(key)

      self.border = border
      self.left, self.right = self.border, self.W-self.border
      self.top, self.bottom = self.border, self.H-self.border
 
   def renderBorder(self):
      for coords in [
            (0, 0, self.W, self.top),
            (0, 0, self.left, self.H),
            (0, self.bottom, self.W, self.H),
            (self.right, 0, self.W, self.H)
            ]:
         pygame.draw.rect(self.canvas, Color.RED, coords)
 
   def reset(self):
      self.canvas.fill((Color.BLACK))
      if self.border > 0:
          self.renderBorder()

   def fill(self, color):
      self.canvas.fill((color))

   def blit(self, container, pos, area=None, flags=0):
      w, h = pos
      pos = (w+self.border, h+self.border)
      if type(container) == Surface:
         self.canvas.blit(container, pos, area=area, special_flags=flags)
      else:
         self.canvas.blit(container.canvas, pos, area=area, special_flags=flags)

      if self.border > 0:
         self.renderBorder()

   def rect(self, color, coords, lw=0):
      pygame.draw.rect(self.canvas, color, coords, lw)

   def line(self, color, start, end, lw=1):
      pygame.draw.line(self.canvas, color, start, end, lw)

def fromTiled(tiles, tileSz):
   R, C, three = tiles.shape
   ret = np.zeros((R, C), dtype=object)
   for r in range(R):
      for c in range(C):
         ret[r, c] = Surface((tileSz, tileSz))
         ret[r, c].fill(tiles[r, c, :])
   return ret

def makeMap(tiles, tileSz):
   R, C = tiles.shape
   surf = Surface((int(R*tileSz), int(C*tileSz)))
   for r in range(R):
      for c in range(C):
         surf.blit(tiles[r, c], (int(r*tileSz), int(c*tileSz)))
   return surf
   
def surfToIso(surf):
   ret = pygame.transform.rotate(surf, 45)
   W, H = ret.get_width(), ret.get_height()
   ret = pygame.transform.scale(ret, (W, H//2))
   return ret

def rotate(x, y, theta):
    pt = np.array([[x], [y]])
    mat = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]])
    ret = np.dot(mat, pt)
    return ret.ravel()

def degrees(rad):
   return 180 * rad / 3.14159265

def rad(degrees):
   return 3.14159265 * degrees / 180.0

def pointToIso(point, tW, tH, W, H):
    x, y = point
    x, y = x-W//2, y+H//2
    x, y = rotate(x, y, rad(-45))
    #x, y = x+(W//2*np.sqrt(2)), y-(H//2*np.sqrt(2))
    y = y / 2
    return x, y

def makeMap(tiles, tileSz):
   R, C = tiles.shape
   surf = Surface((int(R*tileSz), int(C*tileSz)))
   for r in range(R):
      for c in range(C):
         #tile = surfToIso(tiles[r, c])
         tile = tiles[r, c]
         rr, cc = int(r*tileSz), int(c*tileSz)
         #cc, rr = pointToIso((cc, rr), tileSz)
         surf.blit(tile, (rr, cc))
   return surf

class Application:
   def __init__(self):
      self.W, self.H, self.side = 512, 512, 256
      self.appSize = (self.W+self.side, self.H+self.side)

      pygame.init()
      pygame.display.set_caption('Projekt: Godsword')
      self.canvas = pygame.display.set_mode(self.appSize, pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE)
      #self.fonts = RenderUtils.Fonts('resource/dragonslapper.ttf')
      '''

      self.buf = Container(self.appSize)

      sz = 16
      tiles = np.array([[Neon.rand12() for i in range(sz)] for j in range(sz)])
      tileSz = 32
      tiles = fromTiled(tiles, tileSz)
      surf = makeMap(tiles, tileSz)
      #self.canvas.blit(surf, (0, 0))
      surf = surfToIso(surf)
      p1, p2 = (0, 0), (tileSz, tileSz)
      p1 = pointToIso(p1, tileSz, tileSz, self.W, self.H)
      p2 = pointToIso(p2, tileSz, tileSz, self.W, self.H)
      #rect = (*p1, *p2)
      self.canvas.blit(surf, (0, 0))
      pygame.draw.line(self.canvas, (255, 255, 255), p1, p2)

      '''
      while True:
          self.render()

   def render(self):
      start = time.time()
      #self.canvas.blit(self.buf.canvas, (0,0))
      pygame.event.get()
      #pygame.draw.rect(self.canvas, (255,0,0), (0, 0, 50, 50))
      pygame.display.flip()
      #pygame.display.update((0, 0, 50, 50))
      print(1/(time.time()-start))

app = Application()
