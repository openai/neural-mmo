#Various Enums used for handling materials, entity types, etc.
#Data texture pairs are used for enums that require textures.
#These textures are filled in by the Render class at run time.

from pdb import set_trace as T
import numpy as np
import colorsys

def rgb(h):
  h = h.lstrip('#')
  return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def rgbNorm(h):
  h = h.lstrip('#')
  return tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))

def makeColor(idx, h=1, s=1, v=1):
   r, g, b = colorsys.hsv_to_rgb(h, s, v)
   rgbval = tuple(int(255*e) for e in [r, g, b])
   hexval = '%02x%02x%02x' % rgbval
   return Color(str(idx), hexval)

class Color:
    def __init__(self, name, hexVal):
        self.name = name
        self.hex = hexVal
        self.rgb = rgb(hexVal)
        self.norm = rgbNorm(hexVal)
        self.value = self.rgb #Emulate enum

    def packet(self):
        return self.hex

class Color256:
   def make256():
      parh, parv = np.meshgrid(np.linspace(0.075, 1, 16), np.linspace(0.25, 1, 16)[::-1])
      parh, parv = parh.T.ravel(), parv.T.ravel()
      idxs = np.arange(256)
      params = zip(idxs, parh, parv)
      colors = [makeColor(idx, h=h, s=1, v=v) for idx, h, v in params]
      return colors
   colors = make256()

class Color16:
   def make():
      hues   = np.linspace(0, 1, 16)
      idxs   = np.arange(256)
      params = zip(idxs, hues)
      colors = [makeColor(idx, h=h, s=1, v=1) for idx, h in params]
      return colors
   colors = make()

class Tier:
   BLACK    = Color('BLACK', '#000000')
   WOOD     = Color('WOOD', '#784d1d')
   BRONZE   = Color('BRONZE', '#db4508')
   SILVER   = Color('SILVER', '#dedede')
   GOLD     = Color('GOLD', '#ffae00')
   PLATINUM = Color('PLATINUM', '#cd75ff')
   DIAMOND  = Color('DIAMOND', '#00bbbb')
   
class Swatch:
   def colors():
      '''Return list of swatch colors'''
      return

   def rand():
      '''Return random swatch color'''
      all_colors = colors()
      randInd = np.random.randint(0, len(all_colors))
      return all_colors[randInd]


class Neon(Swatch):
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

   TERM     = Color('TERM', '#41ff00')
   MASK     = Color('MASK', '#d67fff')

   def colors():
      return (
              Neon.CYAN, Neon.MINT, Neon.GREEN,
              Neon.BLUE, Neon.PURPLE, Neon.MAGENTA,
              Neon.FUCHSIA, Neon.SPRING, Neon.SKY,
              Neon.RED, Neon.ORANGE, Neon.YELLOW)

class Solid(Swatch):
   BLUE       = Color('BLUE', '#1f77b4')
   ORANGE     = Color('ORANGE', '#ff7f0e')
   GREEN      = Color('GREEN', '#2ca02c')

   RED        = Color('RED',  '#D62728')
   PURPLE     = Color('PURPLE', '#9467bd')
   BROWN      = Color('BROWN', '#8c564b')

   PINK       = Color('PINK', '#e377c2')
   GREY       = Color('GREY', '#7f7f7f')
   CHARTREUSE = Color('CHARTREUSE', '#bcbd22')

   SKY        = Color('SKY', '#17becf')

   def colors():
      return (
              Solid.BLUE, Solid.ORANGE, Solid.GREEN,
              Solid.RED, Solid.PURPLE, Solid.BROWN,
              Solid.PINK, Solid.CHARTREUSE, Solid.SKY,
              Solid.GREY)

class Palette:
   def __init__(self, initial_swatch=Neon):
      self.colors = {}
      for idx, color in enumerate(initial_swatch.colors()):
          self.colors[idx] = color

   def color(self, idx):
      if idx in self.colors:
           return self.colors[idx]

      color = makeColor(idx, h=np.random.rand(), s=1, v=1)
      self.colors[idx] = color
      return color
