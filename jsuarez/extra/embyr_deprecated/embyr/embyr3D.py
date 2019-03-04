from pdb import set_trace as T
import numpy as np
import time
from scipy.misc import imresize, imsave
from enum import Enum

import pygame
from pygame import Surface

from forge.embyr import embyr
from forge.embyr import utils as renderutils
from forge.embyr import render
from forge.embyr.texture import TextureInitializer
from forge.blade.lib.enums import Neon, Color256, Defaults
from forge.blade.action.v2 import Attack
from forge.blade.action import action

from pdb import set_trace as T
import numpy as np

import os
import kivy3
from kivy.app import App
from kivy3 import Scene, Renderer, PerspectiveCamera
from kivy3.loaders import OBJMTLLoader
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget as KivyWidget
from kivy.config import Config
from kivy.graphics import opengl as gl
from kivy.graphics import Mesh as KivyMesh
from kivy3.core.object3d import Object3D
from kivy3.materials import Material
from kivy.core.image import Image
from copy import deepcopy
from forge.embyr.embyr import Application as KivyApp
import pywavefront as pywave
import pytmx
from forge.blade.lib import enums
from forge.embyr.transform import Transform

root = 'forge/embyr/'
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
shaderf = 'tex/default.glsl'
pi = 3.14159265
NANIM = 4

class MaterialMesh(Object3D):
   def __init__(self, mesh, material):
      super().__init__()
      self._mesh = mesh
      self.material = material
      self.mtl = material
      self.bind(pos = self.update_pos)

      self.material.diffuse   = material.diffuse[:3]
      self.material.specular  = material.specular[:3]
      self.material.shininess = material.shininess

   def retexturef(self, texf):
      tex = Image(texf).texture
      self.retexture(self, tex)

   def retexture(self, tex):
      self._mesh.texture = tex

   def update_pos(self, x, y, z):
      self.pos.x = float(x)
      self.pos.y = float(y)
      self.pos.z = float(z)

   def custom_instructions(self):
      yield self.material
      yield self._mesh

class OBJ(MaterialMesh):
   def __init__(self, objs):
       super(MaterialMesh, self).__init__()
       for _, obj in objs.items():
          mesh, mat = obj
          matmesh = MaterialMesh(mesh, mat)
          self.add(matmesh)

   def custom_instructions(self):
       for child in self.children:
          for instr in child.custom_instructions():
             yield instr

class Block(MaterialMesh):
    def __init__(self, tile):
       objs = OBJLoader.loadRaw(root+'block.obj',
             root+'tex/'+tile+'.png')['Cube']
       super().__init__(*objs)
       self.block(tile)

    def block(self, tile):
       if tile == 'lava':
          self.pos.y = -0.5
       elif tile == 'stone':
          self.pos.y = 1
       elif tile == 'water':
          self.pos.y = -0.33

class Ent(OBJ):
    def __init__(self):
       objs = OBJLoader.loadRaw(root+'tex/nn.obj')
       super().__init__(objs)

class Map:
    def __init__(self, scene, path='resource/maps/map1/map.tmx'):
        tiled = self.loadTiled(path)
        self.blocks = self.load(tiled, scene)

    def load(self, tiles, scene, border=8):
        blocks, self.shape = {}, tiles.shape
        self.R, self.C = tiles.shape
        for i in range(border, self.R-border):
           for j in range(border, self.C-border):
              tile = tiles[i, j]

              #Load unit cube
              block = Block(tile)
              scene.add(block)
              blocks[(i, j)] = block

              #NEVER set .pos directly.
              #It won't do anything
              block.pos.x = i
              block.pos.z = j

              block.x0 = block.pos.x
              block.y0 = block.pos.y
              block.z0 = block.pos.z

        return blocks

    def loadTiled(self, fPath):
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

class OBJLoader:
    def formats(name):
       if name == 'T2F_N3F_V3F':
          return [
             (b'v_tc0', 2, 'float'),
             (b'v_normal', 3, 'float'),
             (b'v_pos', 3, 'float')]
       elif name == 'N3F_V3F':
          return [
             (b'v_normal', 3, 'float'),
             (b'v_pos', 3, 'float')]

    def load(objf, texf=None):
       raw = OBJLoader.loadRaw(objf, texf)
       return OBJ(raw)

    def loadRaw(objf, texf=None):
       rets, obj = {}, pywave.Wavefront(objf, collect_faces=True)
       for name, mesh in obj.meshes.items():
          rets[name] = OBJLoader.loadSingle(mesh, texf)
       return rets

    def loadSingle(obj, texf=None):
       assert len(obj.materials) == 1
       material = obj.materials[0]

       if texf is None:
          texf = material.texture.image_name
       tex = Image(texf).texture

       vertices = material.vertices
       nVertex = len(vertices) / material.vertex_size
       indices = np.arange(nVertex).astype(int).tolist()
       fmt = OBJLoader.formats(material.vertex_format)

       kw  = {"vertices": vertices, "indices": indices,
              "fmt": fmt,
              "mode": "triangles",
              'texture':tex
              }

       mesh = KivyMesh(**kw)
       mat = Material(tex)
       return mesh, mat

class Widget(KivyWidget):
   def __init__(self, root, pos=(0, 0), size=(1, 1)):
      super().__init__()
      self.root = root
      w, h = pos
      self.pos_hint = {'left':w, 'top':h}
      self.size_hint = size

       
class RS(Widget):
    def __init__(self, root, **kwargs):
        super().__init__(root)
        self.path = 'forge/embyr/'
        self.glSetup()
        self.camera = PerspectiveCamera(30, 1, 10, 1000)
        #self.transform = Transform()
        self.transform = Transform(
              pos=[8, 20, 8], lookvec=[40, 10, 40])
        
        pos = kwargs.pop('pos', None)
        size   = kwargs.pop('size', None)
        #pos_hint = {'right':ww, 'bottom':hh}
        self.renderer = Renderer(shader_file=self.path+shaderf)
        #self.renderer.pos_hint = pos_hint

        self.scene = Scene()
        #self.renderer.size_hint = (0.5, 0.5)
        self.transform.size = (0, 0)
        self.transform.size_hint = (0, 0)

        self.renderer.render(self.scene, self.camera)
        self.renderer.bind(size=self._adjust_aspect)
        self.renderer.size_hint = size
        self.renderer.pos = pos

        #self.renderer.pos = (256, 256)

        self.root.add_widget(self.renderer)
        self.root.add_widget(self.transform)
        #self._disabled_count = 0
        #self.add_widget(self.renderer)
        #self.add_widget(self.transform)

    def glSetup(self):
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_BACK)

    def add(self, obj):
        self.scene.add(obj)
        self.renderer.add(obj)

    def update(self, dt):
        pos, vec = self.transform.update(dt)
        self.renderer.camera.look_at(vec)
        x, y, z = pos
        self.renderer.camera.pos = (-x, -y, -z)
        return vec

    def _adjust_aspect(self, inst, val):
        rsize = self.renderer.size
        aspect = rsize[0] / float(rsize[1])
        self.renderer.camera.aspect = aspect


