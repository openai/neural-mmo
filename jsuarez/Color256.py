def makeColor(idx, h=1, s=1, v=1):
   r, g, b = colorsys.hsv_to_rgb(h, s, v)
   rgbval = tuple(int(255*e) for e in [r, g, b])
   hexval = '%02x%02x%02x' % rgbval
   return Color(str(idx), hexval)

class Color256:
   def make256():
      parh, parv = np.meshgrid(np.linspace(0.075, 1, 16), np.linspace(0.25, 1, 16)[::-1])
      parh, parv = parh.T.ravel(), parv.T.ravel()
      idxs = np.arange(256)
      params = zip(idxs, parh, parv)
      colors = [makeColor(idx, h=h, s=1, v=v) for idx, h, v in params]
      return colors
   colors = make256()

