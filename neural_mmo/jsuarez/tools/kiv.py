from pdb import set_trace as T
import numpy as np
import pygame
import kivy as kv
from kivy.app import App
from kivy.core.image import ImageData
from kivy.uix.widget import Widget
from kivy.core.image import Image
from kivy.graphics.texture import Texture
from kivy.graphics import Rectangle, Color
from kivy.config import Config
from kivy.clock import Clock
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

def blit(canvas, data, pos):
    data = data.tostring()
    tex  = Texture.create(size=sz, colorfmt="rgb")
    tex.blit_buffer(data, bufferfmt="ubyte", colorfmt="rgb")
    canvas.texture = tex
    canvas.pos = pos

def makeTex():
    #sz = (2048+256, 1024+256)
    sz = (2880, 1800)
    texture = Texture.create(size=sz, colorfmt="rgb")
    arr = np.random.randint(0, 256, (sz[0], sz[1], 3)).astype(np.uint8)
    data = arr.tostring()
    texture.blit_buffer(data, bufferfmt="ubyte", colorfmt="rgb")
    return texture

class Container(Widget):
    def __init__(self, size, **kwargs):
        super(Container, self).__init__(**kwargs)
        self.size = size
        self.W, self.H = size
        with self.canvas:
           self.screen = Rectangle(pos=(0, 0), size=size)    
        self.surf = pygame.Surface((self.W, self.H))
        self.texture = Texture.create(size=size, colorfmt="rgb")

    def reset(self):
      self.fill((Color.BLACK))
      if self.border > 0:
          self.renderBorder()

    def flip(self):
        fast = True
        if fast:
            data = pygame.image.tostring(self.surf, 'RGB', False)
            self.texture.blit_buffer(data, bufferfmt="ubyte", colorfmt="rgb")
        else:
            data = pygame.surfarray.array3d(self.surf)
            data = data.transpose(1, 0, 2).tostring()
            self.texture.blit_buffer(data, bufferfmt="ubyte", colorfmt="rgb")
        self.screen.texture = self.texture

    def renderBorder(self):
      for coords in [
            (0, 0, self.W, self.top),
            (0, 0, self.left, self.H),
            (0, self.bottom, self.W, self.H),
            (self.right, 0, self.W, self.H)
            ]:
         pygame.draw.rect(self.surf, Color.RED, coords)

    def blit(self, container, pos, area=None, flags=0):
      w, h = pos
      pos = (w+self.border, h+self.border)
      if type(container) == Surface:
         self.canvas.blit(container, pos, area=area, special_flags=flags)
      else:
         self.canvas.blit(container.canvas, pos, area=area, special_flags=flags)

      if self.border > 0:
         self.renderBorder()

    def fill(self, color):
      self.surf.fill((color))

    def rect(self, color, coords, lw=0):
        pygame.draw.rect(self.surf, color, coords, lw)

    def line(self, color, start, end, lw=1):
        pygame.draw.line(self.surf, color, start, end, lw)

class Game(Container):
    def __init__(self, size, **kwargs):
        super(Game, self).__init__(size, **kwargs)

    def update(self, t):
        print(1/t)
        sz = 32
        n = 32
        self.rect(Neon.GRAY.value, (0, 0, self.W, self.H))
        for r in range(n):
            for c in range(n):
                if r not in (0, n-1):
                    if c not in (0, n-1):
                        continue
                color = Neon.rand12()
                rect = ((c+1)*sz, (r+1)*sz, sz, sz)
                self.rect(color, rect)
        self.flip()
 
class MyApp(App):
    def build(self):
        size = (2048+256, 1024+256)
        self.title = 'hello'
        #surf = pygame.Surface(size)
        #surf.fill(Neon.RED.value)
        game = Game(size)
        game.update(1)
        #Clock.schedule_interval(game.update, 1.0/.001)
        return game

if __name__ == '__main__':
    MyApp().run()
