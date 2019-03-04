from pdb import set_trace as T

import numpy as np
import time
from kivy.uix.widget import Widget
from kivy.config import Config

rad = 80
sz = 500
pi = 3.14159265

class Transform(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.zoom   = Zoom()
        self.pan    = Pan('left', self.zoom)
        self.rotate = Rotate('right', self.pan)

        self.t = 0
        self.vInit = np.array([0, 0, 1])

        self.add_widget(self.pan)
        self.add_widget(self.zoom)
        self.add_widget(self.rotate)
        self.button = None

    def update(self, dt):
        self.t += dt

        rot_xz, xz_vec, xz_norm, vec = self.rotate(self.vInit)
        vec = self.zoom(vec)
        pos, lookvec = self.pan(vec, rot_xz, xz_vec)

        return lookvec, pos

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

    def clip(self, zoom, exprange=2):
        return np.clip(zoom, 0.5**exprange, 2**exprange)

    def on_touch_down(self, touch):
        if touch.button == 'scrollup':
            self.zoom = self.clip(self.zoom + self.delta)
        elif touch.button == 'scrolldown':
            self.zoom = self.clip(self.zoom - self.delta)

    def __call__(self, vec):
        return vec * self.zoom


class Pan(TouchWidget):
    def __init__(self, button, zoom, **kwargs):
        super().__init__(button, **kwargs)
        self.x, self.y, self.xVol, self.yVol = 0, 0, 0, 0
        self.zoom = zoom

    def on_touch_move(self, touch):
        super().on_touch_move(touch)
        self.xVol = int(self.xVol * self.zoom.zoom)
        self.yVol = int(self.yVol * self.zoom.zoom)

    def __call__(self, vec, rot_xz, xz_vec):
        x =  10 * (self.x + self.xVol) / sz
        z = -10 * (self.y + self.yVol) / sz

        #Horizontal component
        unit_y = np.array([0, 1, 0])
        xx, _, zz = np.cross(rot_xz, unit_y)
        norm = np.sqrt(xx**2 + zz**2)
        xh, zh = x*xx/norm, x*zz/norm

        #Depth component
        xx, _, zz = -xz_vec
        norm = np.sqrt(xx**2 + zz**2)
        xd, zd = z*xx/norm, z*zz/norm

        xx, yy, zz = vec
        vec = [xx+xh+xd, yy, zz+zh+zd]
        lookvec = [-xh-xd, 0, -zh-zd]
        return vec, lookvec

class Rotate(TouchWidget):
    def __init__(self, button, pan, **kwargs):
        super().__init__(button, **kwargs)
        self.pan = pan

    def __call__(self, vec):
        x = (self.x + self.xVol) / sz
        y = (self.y + self.yVol) / sz
        yclip = np.clip(y, -pi/2, 0)

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
        rot_xz = rad*np.dot(xz, vec)
        xx, _, zz = rot_xz
        xz_vec = np.array([xx, 0, zz])

        #Find spherical yz plane rotation
        _, yy, zn = np.dot(yz, vec)
        xz_norm = zn

        #For x, z: shrink to position of spherical rotation
        #For y: use height from spherical rotation
        vec = np.array([xx*xz_norm, -rad*yy, zz*xz_norm])
        return rot_xz, xz_vec, xz_norm, vec

