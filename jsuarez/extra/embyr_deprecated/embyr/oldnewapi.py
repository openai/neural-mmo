from pdb import set_trace as T
import numpy as np

import os
import kivy3
from kivy.app import App
from kivy3 import Scene, Renderer, PerspectiveCamera
from kivy3.loaders import OBJLoader, OBJMTLLoader
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget
from kivy.config import Config
from kivy.graphics import opengl as gl
from kivy.graphics import Mesh as KivyMesh
from kivy3.core.object3d import Object3D
from kivy3.materials import Material
from kivy.core.image import Image
from copy import deepcopy
from embyr import Application
import pywavefront as pywave
import pytmx
import enums

Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

def loadTiled(fPath):
    import pytmx
    tm = pytmx.TiledMap(fPath)
    assert len(tm.layers) == 1
    layer = tm.layers[0]
    W, H = layer.width, layer.height
    tilemap = np.zeros((H, W), dtype=object)
    for w, h, dat in layer.tiles():
       f = dat[0]
       tex = f.split('/')[-1].split('.')[0]
       tilemap[h, w] = tex
    return tilemap

class Pan(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.x, self.y, self.xVol, self.yVol = 0, 0, 0, 0
        self.zoom, self.zoomVol = 1, 0

    def on_touch_down(self, touch):
        self.xStart, self.yStart = touch.pos

    def on_touch_up(self, touch):
        if touch.button == 'left':
            self.xVol, self.yVol= 0, 0
            xEnd, yEnd = touch.pos
            self.x += xEnd - self.xStart
            self.y += yEnd - self.yStart

    def on_touch_move(self, touch):
        if touch.button == 'left':
            xEnd, yEnd = touch.pos
            self.xVol = xEnd - self.xStart
            self.yVol = yEnd - self.yStart

class Rotate(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.x, self.y, self.xVol, self.yVol = 0, 0, 0, 0
        self.zoom, self.zoomVol = 1, 0

    def on_touch_down(self, touch):
        self.xStart, self.yStart = touch.pos

    def on_touch_up(self, touch):
        if touch.button == 'right':
            self.xVol, self.yVol= 0, 0
            xEnd, yEnd = touch.pos
            self.x += xEnd - self.xStart
            self.y += yEnd - self.yStart
            bound = 3.14159*250
            self.y = int(np.clip(self.y, -bound, 0))

    def on_touch_move(self, touch):
        if touch.button == 'right':
            xEnd, yEnd = touch.pos
            self.xVol = xEnd - self.xStart
            self.yVol = yEnd - self.yStart

class Zoom(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.zoom, self.delta = 1, 0.2

    def clipZoom(zoom, exprange=2):
        return np.clip(zoom, 0.5**exprange, 2**exprange)

    def on_touch_down(self, touch):
        print(touch.button)
        if touch.button == 'scrollup':
            self.zoom = Zoom.clipZoom(self.zoom + self.delta)
        if touch.button == 'scrolldown':
            self.zoom = Zoom.clipZoom(self.zoom - self.delta)
           
class MyApp(Application):
    def __init__(self, size):
        super().__init__(size)
        self.title = 'Projekt: Godsword'
        self.pan    = Pan()
        self.rotate = Rotate()
        self.zoom   = Zoom()

        self.loader = OBJMTLLoader()
        self.blocks = dict((mat.value.tex, mat.value) 
              for mat in enums.Material)

        self.vec = np.array([0, 0, 1])
        self.t = 0

        '''
        super(Mesh, self).__init__(**kw)
        self.geometry = geometry
        self.material = material
        self.mtl = self.material  # shortcut for material property
        self.vertex_format = kw.pop("vertex_format", DEFAULT_VERTEX_FORMAT)
        self.create_mesh()
        '''
    def cube(self, tile):
       DEFAULT_VERTEX_FORMAT = [
            (b'v_tc0', 2, 'float'),
            (b'v_normal', 3, 'float'),
            (b'v_pos', 3, 'float')]


       obj = self.blocks[tile].obj
       T()
       obj = pywave.Wavefront('tex/block.obj', collect_faces=True)
       material = obj.materials['grass']
       cube = obj.meshes['Cube']

       vertices = obj.vertices
       faces = cube.faces
       grass = obj.materials['grass']
       dirt  = obj.materials['dirt']
       vertices = grass.vertices + dirt.vertices
       #indices  = np.array(faces).ravel().tolist()
       indices = np.arange(36).astype(int).tolist()
       #vertices = np.array(vertices).ravel().tolist()

       tex = Image('tex/grass.png').texture
       mat = Material(tex)
       kw = {"vertices": vertices, "indices": indices,
              "fmt": DEFAULT_VERTEX_FORMAT, 
              "mode": "triangles",
              'texture':tex
              }
       #if self.material.map:
       #     kw["texture"] = self.material.map

       mesh = KivyMesh(**kw)
       class Meshy(Object3D):
           def __init__(self, mesh, material):
               super().__init__()
               self._mesh = mesh
               self.material = material
               self.mtl = material
               self.vertex_format = DEFAULT_VERTEX_FORMAT
       cube = Meshy(mesh, tex)

       #cube.material = orig.material
       #cube.geometry = orig.geometry
       orig._mesh = cube._mesh
       orig.material = mat
       cube = orig

       #cube = kivy3.Mesh([], material)
       if tile == 'lava':
          cube.pos.y = -0.5
       elif tile == 'stone':
          cube.pos.y = 1
       elif tile == 'grass':
          pass
       elif tile == 'forest':
          pass
       elif tile == 'water':
          cube.pos.y = -0.33

       #cube.material.color = 0., .7, 0.  # green
       #cube.material.diffuse = 0., .7, 0.  # green
       return cube

    def makeMap(self):
        tiles = loadTiled('../Projekt-Godsword/resource/maps/map1/map.tmx')
        n, sz = tiles.shape[0], 1
        for i in range(n):
           for j in range(n):
              tile = tiles[i, j]
              cube = self.cube(tile)
              self.scene.add(cube)

              #NEVER set cube.pos directly. 
              #It won't do anything
              cube.pos.x = i - n//2 + sz//2
              cube.pos.z = j - n//2 + sz//2

    def glSetup(self):
        #gl.glEnable(gl.GL_CULL_FACE)
        #gl.glCullFace(gl.GL_BACK)
        pass
 
    def build(self):
        camera = PerspectiveCamera(30, 1, 1, 1000)
        self.renderer = Renderer()
        self.scene = Scene()
        root = FloatLayout()

        obj = self.loader.load('tex/nn.obj', 'tex/nn.mtl')
        self.scene.add(obj)
        obj.pos.y = 1
        self.makeMap()

        self.renderer.render(self.scene, camera)
        self.renderer.camera.look_at(0, 0, 0)
        root.add_widget(self.renderer)
        root.add_widget(self.pan)
        root.add_widget(self.rotate)
        root.add_widget(self.zoom)
        self.renderer.bind(size=self._adjust_aspect)
        self.loop(self.update)
        return root

    def update(self, t):
        print(1/t)
        self.t += t
        rad = 80
        sz = 500
        pi = 3.14159265
        

        r = self.rotate
        x = r.x + r.xVol
        y = r.y + r.yVol

        x = x / sz
        y = y / sz
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
        rot_xz = rad*np.dot(xz, self.vec)
        xx, _, zz = rot_xz
        xz_vec = np.array([xx, 0, zz])

        #Find spherical yz plane rotation
        _, yy, zn = np.dot(yz, self.vec)
        xz_norm = zn

        #For x, z: shrink to position of spherical rotation
        #For y: use height from spherical rotation
        vec = np.array([xx*xz_norm, -rad*yy, zz*xz_norm])

        #Zoom factor
        zoom = Zoom.clipZoom(self.zoom.zoom)
        vec = vec * zoom

        p = self.pan
        x = p.x + p.xVol
        z = p.y + p.yVol

        x = 10*x / sz
        z = -10*z / sz

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
        vec = np.array([xx+xh+xd, yy, zz+zh+zd])
        self.renderer.camera.look_at([-xh-xd, 0, -zh-zd])
        self.renderer.camera.pos = vec.tolist()

    def _adjust_aspect(self, inst, val):
        rsize = self.renderer.size
        aspect = rsize[0] / float(rsize[1])
        self.renderer.camera.aspect = aspect

if __name__ == '__main__':
    MyApp((2048, 1024)).run()
