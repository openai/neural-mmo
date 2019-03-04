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

from pdb import set_trace as T
from kivy.graphics import Mesh as KivyMesh
from kivy3 import Vector3
from kivy3.core.object3d import Object3D

DEFAULT_VERTEX_FORMAT = [(b'v_pos', 3, 'float'),
            (b'v_normal', 3, 'float'),
            (b'v_tc0', 2, 'float')]


class Mesh(Object3D):

    def __init__(self, geometry, material, **kw):
        super(Mesh, self).__init__(**kw)
        self.geometry = geometry
        self.material = material
        self.mtl = self.material  # shortcut for material property
        self.vertex_format = kw.pop("vertex_format", DEFAULT_VERTEX_FORMAT)
        self.create_mesh()

    def create_mesh(self):
        """ Create real mesh object from the geometry and material """
        vertices = []
        indices = []
        idx = 0
        for face in self.geometry.faces:
            for i, k in enumerate(['a', 'b', 'c']):
                v_idx = getattr(face, k)
                vertex = self.geometry.vertices[v_idx]
                vertices.extend(vertex)
                try:
                    normal = face.vertex_normals[i]
                except IndexError:
                    normal = Vector3([0, 0, 0])
                vertices.extend(normal)
                try:
                    tex_coords = self.geometry.face_vertex_uvs[0][idx]
                    vertices.extend(tex_coords)
                except IndexError:
                    vertices.extend([0, 0])
                indices.append(idx)
                idx += 1
        kw = {"vertices": vertices, "indices": indices,
              "fmt": self.vertex_format, "mode": "triangles"
              }
        if self.material.map:
            kw["texture"] = self.material.map

        self._mesh = KivyMesh(**kw)

    def custom_instructions(self):
        yield self.material
        yield self._mesh
