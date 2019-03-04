from pdb import set_trace as T

import numpy as np
import time
from kivy.uix.widget import Widget
from kivy.config import Config

rad = 80
sz = 500
pi = 3.14159265

class Transform(Widget):
    def __init__(self, 
           pos=[0, 0, 0.01], lookvec=[0, 0, 0], **kwargs):
        super().__init__(**kwargs)
        self.zoom   = Zoom()
        self.pan    = Pan('left', self.zoom)
        self.rotate = Rotate('right')

        self.t = 0
        self.pos0 = np.array(pos)
        self.vec0 = np.array(lookvec)
        self.posn = np.array(pos)
        self.vecn = np.array(lookvec)


        self.add_widget(self.pan)
        self.add_widget(self.zoom)
        self.add_widget(self.rotate)
        self.button = None

    def update(self, dt):
        self.t += dt
        pos, vec = self.pos0, self.vec0

        pos = self.zoom(pos, vec)
        pos, vec = self.rotate(pos, vec)
        pos, vec = self.pan(pos, vec)

        self.posn = pos
        self.vecn = vec
        return pos, vec

        #pos = self.zoom(pos, vec)
        #pos, vec = self.pan(pos, vec)
        #pos = self.rotate(pos, vec)

        return pos, vec

class TouchWidget(Widget):
    def __init__(self, button, **kwargs):
        super().__init__(**kwargs)
        self.x, self.y, self.xVol, self.yVol = 0, 0, 0, 0
        self.button = button
        self.reset = True

    #Currently broken due to race condition?
    #def on_touch_down(self, touch):
    #    if touch.button == self.button: 
    #       self.xStart, self.yStart = touch.pos

    def on_touch_up(self, touch):
        if touch.button == self.button: 
            self.x += self.xVol
            self.y += self.yVol
            self.xVol, self.yVol= 0, 0
            self.reset = True

    def on_touch_move(self, touch):
        if self.reset:
           self.xStart, self.yStart = touch.pos
           self.reset = False

        if touch.button == self.button:
            xEnd, yEnd = touch.pos
            self.xVol = xEnd - self.xStart
            self.yVol = yEnd - self.yStart

class Zoom(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.zoom, self.delta = 1, 0.2

    def clip(self, zoom, exprange=4):
        return np.clip(zoom, 0.5**exprange, 2**exprange)

    def on_touch_down(self, touch):
        if touch.button == 'scrollup':
            self.zoom = self.clip(self.zoom + self.delta)
        elif touch.button == 'scrolldown':
            self.zoom = self.clip(self.zoom - self.delta)

    def __call__(self, pos, vec):
        #return offvec * self.zoom
        pos = vec + self.zoom*(pos - vec)
        return pos

class Pan(TouchWidget):
    def __init__(self, button, zoom, **kwargs):
        super().__init__(button, **kwargs)
        self.x, self.y, self.xVol, self.yVol = 0, 0, 0, 0
        self.zoom = zoom
        self.panx, self.pany = 0, 0

    def on_touch_move(self, touch):
        super().on_touch_move(touch)
        self.xVol = int(self.xVol * self.zoom.zoom)
        self.yVol = int(self.yVol * self.zoom.zoom)

    def __call__(self, pos, vec):
        x = 10 * (self.x + self.xVol) / sz
        z = 10 * (self.y + self.yVol) / sz

        pos0 = pos
        vec0 = vec
        
        #pos = pos0 + np.array([x, 0, 0])
        #vec = vec0 + np.array([x, 0, 0])
        #return pos, vec

        #Depth vector
        xx, _, zz = pos - vec
        norm = np.sqrt(xx**2 + zz**2)
        xx, zz = xx/norm, zz/norm
        vec = np.array([xx, 0, zz])

        #Horizontal component
        unit_y = np.array([0, 1, 0])
        xh, _, zh = np.cross(vec, unit_y)
        xh, zh = x*xh, x*zh
        horiz = np.array([xh, 0, zh])

        #Depth component
        xd, zd = z*xx, z*zz
        depth = np.array([xd, 0, zd])

        delta = horiz + depth
        #delta = np.array([xh, 0, zh])
        #delta = np.array([xd, 0, zd])
        return pos0 + delta, vec0 + delta

class Rotate(TouchWidget):
    def __call__(self, pos, vec):
        x = (self.x + self.xVol) / sz
        y = (self.y + self.yVol) / sz
        #yclip = np.clip(y, -pi/2, 0)
        yclip = -y

        xz_x = np.cos(x)
        xz_z = np.sin(x)

        yz_y = np.cos(yclip)
        yz_z = np.sin(yclip)

        xz = np.array([
            [xz_x, 0, -xz_z],
            [0   , 1,  0   ],
            [xz_z, 0,  xz_x]])

        yz = np.array([
            [1, 0,    0    ],
            [0, yz_y, -yz_z],
            [0, yz_z, yz_y]])

        #Find cylindrical xz plane rotation
        xx, _, zz = np.dot(xz, pos - vec)

        #Find spherical yz plane rotation
        _, yy, _ = np.dot(yz, pos - vec)

        #For x, z: shrink to position of spherical rotation
        #For y: use height from spherical rotation
        #pos = np.array([xx*xz_norm, yy, zz*xz_norm])
        pos = np.array([xx, yy, zz])
        pos = vec + pos
        return pos, vec

