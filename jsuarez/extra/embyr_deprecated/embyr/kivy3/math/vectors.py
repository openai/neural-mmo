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

__all__ = ('Vector2', 'Vector3', 'Vector4', )

import math

from copy import copy


class BaseVector(list):
    """
        BaseVector is actually 4D vector for optimization
    """

    _d = 4  # dimension size
    _indeces = [0, 1, 2, 3]
    _null = [0, 0, 0, 0]
    _coords = {'x': 0, 'y': 1, 'z': 2, 'v': 3}

    def __init__(self, *largs):
        if len(largs) == 1:
            if len(largs[0]) == self._d:
                super(BaseVector, self).__init__(largs[0])
            else:
                raise Exception('Invalid vector')
        else:
            if len(largs) == self._d:
                super(BaseVector, self).__init__(largs)
            else:
                raise Exception('Invalid vector')
        self._change_cb = None

    def set_change_cb(self, cb):
        self._change_cb = cb

    def set_vector(self, v):
        for i in self._indeces:
            self[i] = v[i]

    def __add__(self, other):
        res = copy(self._null)

        if isinstance(other, BaseVector):
            for i in self._indeces:
                res[i] = self[i] + other[i]
        else:
            for i in self._indeces:
                res[i] = self[i] + other
        return self.__class__(res)

    def add(self, other):
        self.set_vector(self + other)

    @classmethod
    def add_vectors(cls, first, second):
        return first + second

    def __sub__(self, other):
        res = copy(self._null)
        if isinstance(other, BaseVector):
            for i in self._indeces:
                res[i] = self[i] - other[i]
        else:
            for i in self._indeces:
                res[i] = self[i] - other
        return self.__class__(res)

    def sub(self, other):
        self.set_vector(self - other)

    @classmethod
    def sub_vectors(cls, first, second):
        return first - second

    def __mul__(self, other):
        res = copy(self._null)
        if isinstance(other, BaseVector):
            for i in self._indeces:
                res[i] = self[i] * float(other[i])
        else:
            for i in self._indeces:
                res[i] = self[i] * float(other)
        return self.__class__(res)

    def multiply(self, other):
        self.set_vector(self * other)

    @classmethod
    def multiply_vectors(cls, first, second):
        return first * second

    def __div__(self, other):
        res = copy(self._null)
        if isinstance(other, BaseVector):
            for i in self._indeces:
                res[i] = self[i] / float(other[i])
        else:
            for i in self._indeces:
                res[i] = self[i] / float(other)
        return self.__class__(res)

    def divide(self, other):
        self.set_vector(self / other)

    @classmethod
    def divide_vectors(cls, first, second):
        return first / second

    def min(self, v):
        for i in self._indeces:
            if v[i] < self[i]:
                self[i] = v[i]

    def max(self, v):
        for i in self._indeces:
            if v[i] > self[i]:
                self[i] = v[i]

    def clamp(self, vmin, vmax):
        """ This function assumes min < max, if this assumption isn't true
            it will not operate correctly
        """
        for i in self._indeces:
            if self[i] < vmin[i]:
                self[i] = vmin[i]
            elif self[i] > vmax[i]:
                self[i] = vmax[i]

    def negate(self):
        self.set_vector(self * -1)

    def dot(self, v):
        dot = 0
        for i in self._indeces:
            dot += v[i] * self[i]
        return dot

    def length_sq(self):
        length_sq = 0
        for i in self._indeces:
            length_sq += self[i] * self[i]
        return length_sq

    def length(self):
        return math.sqrt(self.length_sq())

    def length_manhattan(self):
        res = 0
        for i in self._indeces:
            res += math.fabs(self[i])
        return res

    def normalize(self):
        return self / self.length()

    def lerp(self, v, alpha):
        for i in self._indeces:
            self[i] += (v[i] - self[i]) * alpha

        return self

    def clamp_scalar(self, n, min, max):
        if n < min:
            return min
        if n > max:
            return max
        return n

    def angle(self, v):
        theta = self.dot(v) / (self.length() * v.length())

        return math.acos(self.clamp_scalar(theta, -1, 1))

    angle_to = angle  # alias for three.js back capability

    def distance(self, v):
        d = self - v
        return d.length()

    distance_to = distance

    def distance_to_squared(self, v):
        d = self - v
        return d.length_sq()

    def __getattr__(self, k):
        if k in self._coords:
            return self[self._coords[k]]
        else:
            raise AttributeError

    def __setattr__(self, k, v):
        if k in self._coords:
            if type(v) == int or type(v) == float:
                self[self._coords[k]] = v
                if self._change_cb:
                    self._change_cb(k, v)
        super(BaseVector, self).__setattr__(k, v)


class Vector4(BaseVector):
    pass


class Vector3(BaseVector):
    _d = 3
    _indeces = [0, 1, 2]
    _null = [0, 0, 0]
    _coords = {'x': 0, 'y': 1, 'z': 2}

    def cross(self, v):
        t = copy(self)

        self[0] = t[1] * v[2] - t[2] * v[1]
        self[1] = t[2] * v[0] - t[0] * v[2]
        self[2] = t[0] * v[1] - t[1] * v[0]

    @classmethod
    def cross_vectors(cls):
        pass


class Vector2(BaseVector):
    _d = 2
    _indeces = [0, 1]
    _null = [0, 0]
    _coords = {'x': 0, 'y': 1}
