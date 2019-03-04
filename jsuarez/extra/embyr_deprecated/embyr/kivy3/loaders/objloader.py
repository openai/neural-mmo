"""
The MIT License (MIT)

Copyright (c) 2013 Niko Skrypnik

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

"""
Loaders for Wavefront format .obj files
=============

"""
from pdb import set_trace as T

import os
from .loader import BaseLoader
from kivy.core.image import Image
from kivy3 import Object3D, Mesh, Material, Vector2
from kivy3.core.geometry import Geometry
from kivy3.core.face3 import Face3


class WaveObject(object):
    """ This class contains top level mesh object information like vertices,
        normals, texcoords and faces
    """

    _mtl_map = {"Ka": "color", "Kd": "diffuse", "Ks": "specular",
                "Ns": "shininess", "Tr": "transparency",
                "d": "transparency", "map_Kd": "map"
                }

    def __init__(self, loader, name=''):
        self.name = name
        self.faces = []
        self.loader = loader
        self.mtl_name = None

    def convert_to_mesh(self, vertex_format=None):
        """Converts data gotten from the .obj definition
        file and create Kivy3 Mesh object which may be used
        for drawing object in the scene
        """

        geometry = Geometry()
        material = Material()
        mtl_dirname = os.path.abspath(os.path.dirname(self.loader.mtl_source))

        v_idx = 0
        # create geometry for mesh
        for f in self.faces:
            verts = f[0]
            norms = f[1]
            tcs = f[2]
            face3 = Face3(0, 0, 0)
            for i, e in enumerate(['a', 'b', 'c']):
                #get normal components
                n = (0.0, 0.0, 0.0)
                if norms[i] != -1:
                    n = self.loader.normals[norms[i] - 1]
                face3.vertex_normals.append(n)

                #get vertex components
                v = self.loader.vertices[verts[i] - 1]
                geometry.vertices.append(v)
                setattr(face3, e, v_idx)
                v_idx += 1

                #get texture coordinate components
                t = (0.0, 0.0)
                if tcs[i] != -1:
                    t = self.loader.texcoords[tcs[i] - 1]
                tc = Vector2(t[0], 1. - t[1])
                geometry.face_vertex_uvs[0].append(tc)

            geometry.faces.append(face3)

        # apply material for object
        if self.mtl_name in self.loader.mtl_contents:
            raw_material = self.loader.mtl_contents[self.mtl_name]
            for k, v in raw_material.items():
                _k = self._mtl_map.get(k, None)
                if k in ["map_Kd", ]:
                    map_path = os.path.join(mtl_dirname, v[0])
                    tex = Image(map_path).texture
                    material.map = tex
                    continue
                if _k:
                    if len(v) == 1:
                        v = float(v[0])
                        if k == 'Tr':
                            v = 1. - v
                        setattr(material, _k, v)
                    else:
                        v = list(map(lambda x: float(x), v))
                        setattr(material, _k, v)
        mesh = Mesh(geometry, material)
        return mesh


class OBJLoader(BaseLoader):

    def __init__(self, **kw):
        super(OBJLoader, self).__init__(**kw)
        self.mtl_source = None  # source of MTL
        self.mtl_contents = {}  # should be filled in load_mtl

    def load_mtl(self):
        if not os.path.exists(self.mtl_source):
            #TODO show warning about materials file is not found
            return
        for line in open(self.mtl_source, "r"):
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'newmtl':
                mtl = self.mtl_contents[values[1]] = {}
                continue
            elif mtl is None:
                raise ValueError("mtl doesn't start with newmtl statement")
            mtl[values[0]] = values[1:]

    def _load_meshes(self):

        wvobj = WaveObject(self)
        self.vertices = []
        self.normals = []
        self.texcoords = []
        faces_section = False

        for line in open(self.source, "r"):
            if line.startswith('#'):
                continue
            if line.startswith('s'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'o' or values[0] == 'g':
                wvobj.name = values[1]
            elif values[0] == 'mtllib':
                if not self.mtl_source:
                    _obj_dir = os.path.abspath(os.path.dirname(self.source))
                    self.mtl_source = os.path.join(_obj_dir, values[1])
                    self.load_mtl()
            elif values[0] == 'usemtl':
                wvobj.mtl_name = values[1]
            elif values[0] == 'v':
                if faces_section:
                    # here we yield new mesh object
                    faces_section = False
                    yield wvobj
                    wvobj = WaveObject(self)
                v = list(map(float, values[1:4]))
                if self.swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if self.swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(list(map(float, values[1:3])))
            elif values[0] == 'f':
                if not faces_section:
                    faces_section = True
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(-1)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(-1)
                wvobj.faces.append((face, norms, texcoords))
        yield wvobj

    def load(self, source, **kw):
        self.swapyz = kw.pop("swapyz", False)
        return super(OBJLoader, self).load(source, **kw)

    def parse(self):

        obj = Object3D()

        for wvobj in self._load_meshes():
            obj.add(wvobj.convert_to_mesh())

        return obj


class OBJMTLLoader(OBJLoader):
    """ This subclass of Wafefront format files loader
    which allows to use custom MTL file, but not the one is
    defined in .obj file
    """

    def load(self, source, mtl_source, **kw):
        self.mtl_source = mtl_source
        self.load_mtl()
        return super(OBJMTLLoader, self).load(source, **kw)
