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

from kivy3 import Vector3
from kivy3.core.geometry import Geometry
from kivy3.core.face3 import Face3


class BoxGeometry(Geometry):

    _cube_vertices = [(-1, 1, -1), (1, 1, -1),
                      (1, -1, -1), (-1, -1, -1),
                      (-1, 1, 1), (1, 1, 1),
                      (1, -1, 1), (-1, -1, 1),
                      ]

    _cube_faces = [(0, 1, 2), (0, 2, 3), (3, 2, 6),
                   (3, 6, 7), (7, 6, 5), (7, 5, 4),
                   (4, 5, 1), (4, 1, 0), (4, 0, 3),
                   (7, 4, 3), (5, 1, 2), (6, 5, 2)
                   ]

    _cube_normals = [(0, 0, 1), (-1, 0, 0), (0, 0, -1),
                     (1, 0, 0), (0, 1, 0), (0, -1, 0)
                     ]

    def __init__(self, width, height, depth, **kw):
        name = kw.pop('name', '')
        super(BoxGeometry, self).__init__(name)
        self.width_segment = kw.pop('width_segment', 1)
        self.height_segment = kw.pop('height_segment', 1)
        self.depth_segment = kw.pop('depth_segment', 1)

        self.w = width
        self.h = height
        self.d = depth

        self._build_box()

    def _build_box(self):

        for v in self._cube_vertices:
            v = Vector3(0.5 * v[0] * self.w,
                        0.5 * v[1] * self.h,
                        0.5 * v[2] * self.d)
            self.vertices.append(v)

        n_idx = 0
        for f in self._cube_faces:
            face3 = Face3(*f)
            normal = self._cube_normals[n_idx / 2]
            face3.vertex_normals = [normal, normal, normal]
            n_idx += 1
            self.faces.append(face3)
