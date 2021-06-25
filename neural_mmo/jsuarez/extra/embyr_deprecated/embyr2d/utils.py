from scipy.misc import imread
import pygame
import time

from forge.blade.lib.utils import EDA
from forge.blade.lib.enums import Neon 

def readRGB(path):
   return imread(path)[:, :, :3]

def pgRead(path, mask=None):
   try:
      img = readRGB(path)
   except FileNotFoundError:
      return None

   img = pygame.pixelcopy.make_surface(img)
   if mask is not None:
      img.set_colorkey(mask)

   #For some reason, pygame loads images transformed
   img = pygame.transform.flip(img, True, False)
   img = pygame.transform.rotate(img, 90)
   return img

class Font:
   def render(txt, size, color=Neon.GOLD.rgb):
      pass
      #return pygame.font.Font('freesansbold.tt', size).render(text, 1, color)

class Fonts:
   def __init__(self, font='freesansbold.ttf'):
      sizes=(9, 12, 18, 24, 28, 32, 36)
      fonts = [pygame.font.Font(font, sz) for sz in sizes]
      self.tiny, self.small, self.normal, self.large, self.Large, self.huge, self.Huge = fonts

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
