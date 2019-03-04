from pdb import set_trace as T
import pygame, time
import numpy as np

from forge.embyr import embyr
from forge.embyr.modules import *

class Client(embyr.Container):
    def __init__(self, view, size, realm, step, dims, nAnim, **kwargs):
        super().__init__(size, **kwargs)
        self.W, self.H, self.side = dims
        self.realm, self.step = realm, step
        self.view, self.nAnim = view, nAnim

        offset = 16 * 8
        self.x, self.y, self.xVol, self.yVol = -offset, offset, 0, 0
        self.zoom, self.zoomVol = 1, 0

        self.count = 0
        self.frame = 0
        self.frames = []
        self.vals = []
        self.init = time.time()

    def setup(self):
       surf = self.view.renderTitle()
       self.blit(surf, (0, 0))
       self.frame += 1
       self.flip()

    def render(self, t):
        if self.frame == 0:
           return self.setup()
        if self.frame == 1:
           time.sleep(2.5)
        self.update()

    def update(self):
        self.writeFrame()
        self.trans = self.renderOffsets(self.H, self.H)
        keyframe = self.count == 0
        if keyframe:
           self.step()

        self.surf = self.view.render(self.realm, self.trans, keyframe)

        self.count = (self.count + 1) % (self.nAnim+1)
        self.blit(self.surf, (0,0))
        self.flip()
        self.frame += 1

    def writeFrame(self):
      NFRAMES=1800
      return
      if self.frame < NFRAMES:
         print('Frame: ', len(self.frames))
         frame = pygame.surfarray.array3d(pygame.transform.rotate(self.surf, 90))
         frame = np.fliplr(frame)
         #frame = frame[:1024, 256:256+1024]
         frame = frame[:1024, 1024+256:1024+256+1024]
         self.frames.append(frame)
         #pygame.image.save(self.screen, 'resource/data/lrframe'+str(self.frame)+'.png')
      elif self.frame == NFRAMES:
         import imageio
         print('Saving MP4...')
         imageio.mimwrite('swordfrag.mp4', self.frames, fps = 30)
         print('Saved')

    def clipZoom(self, zoom):
        return np.clip(zoom, 1.0, 8.0)

    def renderOffsets(self, W, H):
        #Scale
        zoom = self.clipZoom(self.zoom + self.zoomVol)
        scaleX, scaleY = int(W*zoom), int(H*zoom)

        #Translate
        deltaX = self.x + self.xVol - scaleX/2 + W/2
        deltaY = -self.y - self.yVol - scaleY/2 + H/2
        return scaleX, scaleY, deltaX, deltaY

    def on_touch_down(self, touch):
        self.xStart, self.yStart = touch.pos

    def on_touch_up(self, touch):
        if touch.button == 'left':
            self.xVol, self.yVol= 0, 0
            xEnd, yEnd = touch.pos
            self.x += xEnd - self.xStart
            self.y += yEnd - self.yStart
        elif touch.button == 'right':
            self.zoom = self.clipZoom(self.zoom + self.zoomVol)
            self.zoomVol = 0

    def on_touch_move(self, touch):
        if touch.button == 'left':
            xEnd, yEnd = touch.pos
            self.xVol = xEnd - self.xStart
            self.yVol = yEnd - self.yStart
        elif touch.button == 'right':
            xEnd, yEnd = touch.pos
            delta = (xEnd - self.xStart)/2 - (yEnd - self.yStart)/2
            self.zoomVol = delta/100

    def on_key_down(self, *args):
        text = args[3]
        if text == 'i':
            #Toggle isometric
            trans = self.renderOffsets(self.H, self.H)
            self.view.toggleEnv(trans)
        elif text == 'p':
            T()
        elif text == '[':
            self.view.leftScreenshot()
        else:
            #Toggle overlay
            self.view.key(text)

class Application(embyr.Application):
   def __init__(self, size, realm, step, conf):
       super().__init__(size)
       self.realm, self.step = realm, step
       self.conf = conf

   def build(self):
        W, H, side = self.W, self.H, 256
        self.appSize = (self.W, self.H)

        self.title = 'Projekt: Godsword'
        dims = (self.W-side, self.H-side, side)
        canvas = Canvas(self.appSize, self.realm, dims, self.conf)
        self.client = Client(canvas, self.appSize, self.realm,
              self.step, dims, NANIM)

        self.loop(self.client.render)
        return self.client
