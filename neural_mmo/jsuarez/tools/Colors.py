from pdb import set_trace as T
from matplotlib import pyplot as plt
import numpy as np

def hex2rgb(h):
   h = h.lstrip('#')
   return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

class Neon:
   RED      = '#ff0000'
   ORANGE   = '#ff8000'
   YELLOW   = '#ffff00'

   GREEN    = '#00ff00'
   MINT     = '#00ff80'
   CYAN     = '#00ffff'

   BLUE     = '#0000ff'
   PURPLE   = '#8000ff'
   MAGENTA  = '#ff00ff'

   WHITE    = '#ffffff'
   GRAY     =  '#666666'
   BLACK    = '#000000'

   BLOOD    = '#bb0000'
   BROWN    = '#7a3402'
   GOLD     = '#eec600'
   SILVER   = '#b8b8b8'

   FUCHSIA  = '#ff0080'
   SPRING   = '#80ff80'
   SKY      = '#0080ff'
   TERM     = '#41ff00'

if __name__ == '__main__':
   colors = np.array([
         [Neon.RED, Neon.ORANGE, Neon.YELLOW, Neon.BLOOD, Neon.FUCHSIA],
         [Neon.GREEN, Neon.MINT, Neon.CYAN, Neon.BROWN, Neon.SPRING],
         [Neon.BLUE, Neon.PURPLE, Neon.MAGENTA, Neon.GOLD, Neon.SKY],
         [Neon.BLACK, Neon.GRAY, Neon.WHITE, Neon.SILVER, Neon.TERM]])


   sz = 64
   R, C = colors.shape
   img = np.zeros((R*sz, C*sz, 3), dtype=np.uint8)
   for r in range(R):
      for c in range(C):
         color = hex2rgb(colors[r][c])
         tile = np.zeros((sz, sz, 3)) + color
         img[r*sz:(r+1)*sz, c*sz:(c+1)*sz, :] = tile

   plt.imshow(img)
   plt.show()
