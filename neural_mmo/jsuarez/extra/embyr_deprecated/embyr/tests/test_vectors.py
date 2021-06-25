
import unittest
import math

from kivy3 import Vector3, Vector4, Vector2 

# good values for vector 3, 4, 12, 84


class Vector3Test(unittest.TestCase):

    def test_create(self):
        v = Vector3(1, 2, 3)
        self.assertEquals(v[0], 1)
        self.assertEquals(v[1], 2)
        self.assertEquals(v[2], 3)
        v = Vector3([4, 5, 6])
        self.assertEquals(v[0], 4)
        self.assertEquals(v[1], 5)
        self.assertEquals(v[2], 6)
        try:
            Vector3(1, 2, 3, 4)
            assert False, "This shold not reached"
        except:
            pass
        try:
            Vector3([3, 4, 2, 1])
            assert False, "This shold not reached"
        except:
            pass

    def test_add(self):
        v1 = Vector3(1, 2, 3)
        v2 = Vector3(4, 5, 6)
        v = v1 + v2
        self.assertEqual(v, [5, 7, 9])
        v1.add(v2)
        self.assertEqual(v1, [5, 7, 9])
        self.assertEqual(v + 2, [7, 9, 11])

    def test_sub(self):
        v1 = Vector3(4, 5, 6)
        v2 = Vector3(1, 2, 3)
        v = v1 - v2
        self.assertEqual(v, [3, 3, 3])
        v1.sub(v2)
        self.assertEqual(v1, [3, 3, 3])
        self.assertEqual(v - 3, [0, 0, 0])

    def test_multiply(self):
        v1 = Vector3(5, 6, 7)
        v2 = Vector3(2, 2, 2)
        self.assertEqual(v1 * v2, [10., 12., 14.])
        v1.multiply(v2)
        self.assertEqual(v1, [10., 12., 14.])

    def test_divide(self):
        v1 = Vector3(6, 4, 8)
        v2 = Vector3(2, 2, 2)
        self.assertEqual(v1 / v2, [3., 2., 4.])
        v1.divide(v2)
        self.assertEqual(v1, [3., 2., 4.])

    def test_minmax(self):
        v = Vector3(6, 7, 4)
        v1 = Vector3(3, 5, 8)
        v.min(v1)
        self.assertEqual(v, [3, 5, 4])
        v2 = Vector3(1, 7, 6)
        v.max(v2)
        self.assertEqual(v, [3, 7, 6])

    def test_clamp(self):
        v1 = Vector3(1, 2, 3)
        v2 = Vector3(3, 4, 6)
        v = Vector3(0, 5, 4)
        v.clamp(v1, v2)
        self.assertEqual(v, [1, 4, 4])

    def test_negate(self):
        v = Vector3(2, 2, 2)
        v.negate()
        self.assertEqual(v, [-2, -2, -2])

    def test_length(self):
        v = Vector3(3, 12, 4)
        v = Vector3(12, 4, 3)

        self.assertEqual(v.length(), 13)
        self.assertEqual(v.length_sq(), 13*13)

    def test_angle(self):
        v1 = Vector3(0, 0, 1)
        v2 = Vector3(0, 1, 0)
        angle = v1.angle(v2)
        self.assertEqual(math.degrees(angle), 90.0)
        v1 = Vector3(0, 0, 1)
        v2 = Vector3(0, 0, -1)
        angle = v1.angle(v2)
        self.assertEqual(math.degrees(angle), 180.0)

    def test_distance(self):
        v1 = Vector3(2, 1, 6)
        v2 = Vector3(2, 5, 6)

        self.assertEqual(v1.distance(v2), 4)

    def test_attributes(self):
        v = Vector3(0, 0, 0)
        v.x = 4
        self.assertEqual(v[0], v.x)
        self.assertEqual(v[0], 4)

        v.z = 6
        self.assertEqual(v[2], v.z)
        self.assertEqual(v[2], 6)

        try:
            t = v.v
            assert False, "executing of this string is error"
        except AttributeError:
            pass


class Vector2Test(unittest.TestCase):

    def test_create(self):
        v = Vector2(1, 2)
        try:
            v = Vector2(1, 2, 3)
            assert False, "This should not be normally reached"
        except:
            pass # test is passed normally

    def test_attrbutes(self):
        v = Vector2(0, 0)
        v.x = 4
        self.assertEqual(v[0], v.x)
        self.assertEqual(v[0], 4)

        v.y = 7
        self.assertEqual(v[1], v.y)
        self.assertEqual(v[1], 7)

        try:
            t = v.z
            assert False, "executing of this string is error"
        except AttributeError:
            pass


if __name__ == '__main__':
    unittest.main()
