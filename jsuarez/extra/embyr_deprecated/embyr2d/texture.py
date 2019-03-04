from pdb import set_trace as T
from forge.embyr import utils as renderutils
from forge.embyr import embyr

from forge.blade.lib import enums
from forge.blade.lib.enums import Neon

class TextureInitializer():
   def __init__(self, sz, root='resource/', mipLevels=[16, 32, 64, 96, 128], barSz=2):
      self.tileSz = sz
      self.width = sz
      self.barSz = barSz
      self.root = root
      self.mipLevels = mipLevels

      #self.material  = RenderUtils.readRGB(root+'Material/textures.png')
      self.material  = self.textureTiles(mipLevels)
      self.entity    = self.textureEnum(enums.Entity,
            root+'Entity/', mask=Neon.MASK.rgb)
      proj = {
              'melee':renderutils.pgRead(root+'Action/melee.png', mask=Neon.MASK.rgb),
              'range':renderutils.pgRead(root+'Action/range.png', mask=Neon.MASK.rgb),
              'mage':renderutils.pgRead(root+'Action/mage.png', mask=Neon.MASK.rgb),
              }
      action = {
              'melee':renderutils.pgRead(root+'Action/meleeattack.png', mask=Neon.MASK.rgb),
              'range':renderutils.pgRead(root+'Action/rangeattack.png', mask=Neon.MASK.rgb),
              'mage':renderutils.pgRead(root+'Action/mageattack.png', mask=Neon.MASK.rgb),
              }
 
      self.action = embyr.mips(action, mipLevels)

      #self.action    = self.textureEnum(Enums.Entity,
      #      root+'Action/', mask=Color.MASK)

   def textureTiles(self, mipLevels):
      reverse = {}
      for mat in enums.Material:
         mat = mat.value
         texCoords = mat.tex
         mat.tex = renderutils.pgRead(self.root + '/tiles/' + mat.tex + '.png')
         reverse[mat.index] = mat.tex
      return embyr.mips(reverse, self.mipLevels)

   def textureEnum(self, enum, path, mask=None, suffix='.png'):
      reverse = {}
      for color in enums.Neon.color12():
         name = color.name
         texPath = path + 'neural' + name + suffix
         reverse[name] = renderutils.pgRead(texPath, mask=mask)
      for name in range(0, 256):
         name = str(name)
         texPath = path + 'neural' + name + suffix
         reverse[name] = renderutils.pgRead(texPath, mask=mask)
      return embyr.mips(reverse, self.mipLevels)


