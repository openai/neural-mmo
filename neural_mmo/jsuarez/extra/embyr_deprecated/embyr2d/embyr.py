#Author: Joseph Suarez

from pdb import set_trace as T
from scipy.misc import imread
import numpy as np
import time

import pygame
from pygame.surface import Surface

import kivy as kv
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics.texture import Texture as kvTex
from kivy.graphics import Rectangle
from kivy.config import Config
from kivy.clock import Clock
from kivy.core.window import Window

Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

#Wrapper for all Embyr programs
#Mandatory, required by kivy
class Application(kv.app.App):
   def __init__(self, size):
      super().__init__()
      pygame.init()
      self.size = size
      self.W, self.H = size
      Window.size = (self.W//2, self.H//2)
      self.scale = 1.5

   #Run a thunk forever
   def loop(self, func, interval=0):
      Clock.schedule_interval(func, interval)

   #Run a thunk once 
   def once(self, func, delay=0):
      Clock.schedule_once(func, delay)

#Wrapper for all draw canvas objects. 
#Provides a seamless interface between
#pygame surfaces and kivy graphics using
#a fast flip() buffer transfer
#Expected top performance on a laptop:
#~60fps at 1080p or ~24fps at 4k
#key: alpha mask color
#border: size of frame around canvas
class Container(kv.uix.widget.Widget):
   def __init__(self, size, key=None, border=0):
      super().__init__()
      self.W, self.H = size
      self.size, self.border = size, border
      self.surf = pygame.Surface((self.W, self.H))
      #self.surf = pygame.Surface((int(1.5*self.W), int(1.5*self.H)))
      self.texture = kvTex.create(size=size, colorfmt="rgb")

      self.left, self.right = self.border, self.W-self.border
      self.top, self.bottom = self.border, self.H-self.border
      self.scale = 1.5

      if key is not None:
         self.surf.set_colorkey(key)
      with self.canvas:
         self.screen = Rectangle(pos=(0, 0), size=size)
      if hasattr(self, 'on_key_down'):
         Window.bind(on_key_down=self.on_key_down)

   #Clear the canvas
   def reset(self):
      self.fill(Neon.BLACK.rgb)
      if self.border > 0:
         self.renderBorder()

   #Only use in the top level canvas
   #Transfers a pygame data buffer to a kivyv texture
   def flip(self):
      W, H = self.surf.get_width(), self.surf.get_height()
      #surf = pygame.transform.scale(self.surf, (int(W*self.scale), int(H*self.scale)))
      data = pygame.image.tostring(self.surf, 'RGB', True)
      self.texture.blit_buffer(data, bufferfmt="ubyte", colorfmt="rgb")
      self.screen.texture = self.texture

   #Render a frame around the canvas
   def renderBorder(self):
      if self.border == 0:
         return
      for coords in [
            (0, 0, self.W, self.top),
            (0, 0, self.left, self.H),
            (0, self.bottom, self.W, self.H),
            (self.right, 0, self.W, self.H)
         ]:
         pygame.draw.rect(self.surf, Neon.RED.rgb, coords)

   ### Blitting and drawing wrappers ###

   def blit(self, container, pos, area=None, flags=0):
      pos = (pos[0]+self.border, pos[1]+self.border)
      self.surf.blit(container, pos, area=area, special_flags=flags)
      self.renderBorder()

   def fill(self, color):
      self.surf.fill(color)

   def rect(self, color, coords, lw=0):
      pygame.draw.rect(self.surf, color, coords, lw)

   def line(self, color, start, end, lw=1):
      pygame.draw.line(self.surf, color, start, end, lw)

#Return the visible portion of a tiled environment
def mapCrop(screen, env, txSz, trans):
   offs, txSz = offsets(screen, *env.shape[:2], txSz, trans)
   minH, maxH, minW, maxW = offs
   return env[minH:maxH, minW:maxW], txSz

#Array index offsets based on a transformation
def offsets(screen, R, C, txSz, trans=None):
   W, H = screen.W, screen.H
   sx, sy, tx, ty = W, H, 0, 0
   if trans is not None:
      sx, sy, tx, ty = trans
   ratio = sx / W #assume uniform transform
   txSz = txSz * ratio

   nW = int(W / txSz)
   nH = int(H / txSz)
   minW = -int(min(tx / txSz, 0))
   minH = -int(min(ty / txSz, 0))

   maxW = min(minW + nW, C)
   maxH = min(minH + nH, H)
   return (minH, maxH, minW, maxW), txSz

#Make a set of mipmaps for a dict of textures
#with sizes specivied by mipLevels
def mips(texs, mipLevels):
   rets = {}
   for sz in mipLevels:
      if sz not in rets:
         rets[sz] = {}
      for k, tx in texs.items():
         rets[sz][k] = pygame.transform.scale(tx, (sz, sz))
   return rets
   
#Mipmaps for a single texture
def mip(tex, mipLevels):
   return dict((sz, pygame.transform.scale(tex, (sz, sz))) for sz in mipLevels)

#Get the mip map "closest" to the specified resolution
def mipKey(txSz, mipLevels):
   inds = np.array(mipLevels) >= txSz
   return mipLevels[np.argmax(inds)]

#Render a full tiled map to the given screen (Container)
#env specifies an array of texture indices (requires a tex dict)
#or an array of colors otherwise.
#iso can be used to render square maps in isometric perspective
def renderMap(screen, env, txSz, tex=None, iso=False):
   shape = env.shape
   H, W = shape[:2]
   for h in range(H):
      for w in range(W):
         if tex is None:
            ww, hh = tileCoords(w, h, W, H, txSz, iso)
            screen.rect(env[h, w], (ww, hh, txSz, txSz))
         else:
            mat = tex[env[h, w]]
            renderTile(screen, mat, w, h, txSz, iso)

#Convert a tile to isometric
def tileToIso(tex, txSz):
   tex.set_colorkey(Neon.BLACK.rgb)
   tex = pygame.transform.rotate(tex, -45)
   ww, hh = tex.get_width(), tex.get_height()//2
   tex = pygame.transform.scale(tex, (ww, hh))
   return tex

#coords of a tile in ortho perspective
def cartCoords(w, h, txSz):
   return int(w*txSz), int(h*txSz)

#coords of a tile in isometric perspective
def isoCoords(w, h, W, H, txSz):
   nW, nH = W//txSz, H//txSz
   ww = nW//2 + (w - h) / 2
   hh = nH//4 + (w + h) / 4
   w, h = cartCoords(ww, hh, txSz)
   return w, h

#coords of a tile in ortho/isometric perspective
def tileCoords(w, h, W, H, txSz, iso):
   if iso:
      return isoCoords(w, h, W, H, txSz)
   else:
      return cartCoords(w, h, txSz)

#Isometric render a tile
def renderIso(screen, tex, w, h, txSz):
   tex = tileToIso(tex, txSz)
   w, h = isoCoords(w, h, screen.W, screen.H, txSz)
   screen.blit(tex, (w, h))

#Ortho render a tile
def renderCart(screen, tex, w, h, txSz):
   w, h = cartCoords(w, h, txSz)
   screen.blit(tex, (w, h))

#Ortho/isometric render a tile
def renderTile(screen, tex, w, h, txSz, iso):
   if iso:
      renderIso(screen, tex, w, h, txSz)
   else:
      renderCart(screen, tex, w, h, txSz)

#Basic color class with different color schemes
#usable almost anywhere (provides hex/rgb255/rgb1 vals)
class Color:
   def __init__(self, name, hexVal):
      self.name = name
      self.hex = hexVal
      self.rgb = rgb(hexVal)
      self.norm = rgbNorm(hexVal)
      self.value = self.rgb #Emulate enum

def rgb(h):
   h = h.lstrip('#')
   return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def rgbNorm(h):
   h = h.lstrip('#')
   return tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))

#Neon color pallete + a few common game colors.
#Why would you need anything else?
class Neon:
   RED      = Color('RED', '#ff0000')
   ORANGE   = Color('ORANGE', '#ff8000')
   YELLOW   = Color('YELLOW', '#ffff00')

   GREEN    = Color('GREEN', '#00ff00')
   MINT     = Color('MINT', '#00ff80')
   CYAN     = Color('CYAN', '#00ffff')

   BLUE     = Color('BLUE', '#0000ff')
   PURPLE   = Color('PURPLE', '#8000ff')
   MAGENTA  = Color('MAGENTA', '#ff00ff')

   FUCHSIA  = Color('FUCHSIA', '#ff0080')
   SPRING   = Color('SPRING', '#80ff80')
   SKY      = Color('SKY', '#0080ff')

   WHITE    = Color('WHITE', '#ffffff')
   GRAY     = Color('GRAY', '#666666')
   BLACK    = Color('BLACK', '#000000')

   BLOOD    = Color('BLOOD', '#bb0000')
   BROWN    = Color('BROWN', '#7a3402')
   GOLD     = Color('GOLD', '#eec600')
   SILVER   = Color('SILVER', '#b8b8b8')

   #Hacker green
   TERM     = Color('TERM', '#41ff00')

   #Purple alpha
   MASK     = Color('MASK', '#d67fff')

   #Get the 12 neon colors
   def color12():
      return (
            Neon.RED, Neon.ORANGE, Neon.YELLOW,
            Neon.GREEN, Neon.MINT, Neon.CYAN,
            Neon.BLUE, Neon.PURPLE, Neon.MAGENTA,
            Neon.FUCHSIA, Neon.SPRING, Neon.SKY)

   #Get a random neon color
   def rand12():
      twelveColor = color12()
      randInd = np.random.randint(0, len(twelveColor))
      return twelveColor[randInd]

#Pygame Surface initializer from file
def Texture(path, mask=None):
   try:
      img = imread(path)[:, :, :3]
   except FileNotFoundError:
      raise

   img = pygame.pixelcopy.make_surface(img)
   if mask is not None:
      img.set_colorkey(mask)

   #For some reason, pygame loads images transformed
   img = pygame.transform.flip(img, True, False)
   img = pygame.transform.rotate(img, 90)
   return img

#Pygame blank Surface initializer
def Canvas(size, mask=None):
   img = pygame.Surface(size)
   if mask is not None:
      img.set_colorkey(mask)
   return img

#Pygame font wrapper
class Font:
   def __init__(self, size, name='freesansbold.ttf'):
      self.font = pygame.font.Font(name, size)

   def render(self, size, color):
      return self.font.render(size, 1, color)

#Exponentially decaying FPS tracker
class FPSTracker:
   def __init__(self):
      self.start = time.time()
      self.eda = EDA(k=0.95)
      self.fpsVal = 0.0

   def update(self):
      tick = time.time() - self.start
      self.eda.update(1.0/tick)
      self.fpsVal = self.eda.eda
      self.start = time.time()

   @property
   def fps(self):
      return str(self.fpsVal)[:5]

#Exponentially decaying average
class EDA():
   def __init__(self, k=0.9):
      self.k = k
      self.eda = None

   def update(self, x):
      if self.eda is None:
         self.eda = x
         return
      self.eda = (1-self.k)*x + self.k*self.eda
