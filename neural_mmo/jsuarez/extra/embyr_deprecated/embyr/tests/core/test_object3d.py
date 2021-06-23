
import unittest
from kivy3 import Object3D
from tests.utils import Sandbox


class DummyObject:
    pass


class Object3DTest(unittest.TestCase):

    def setUp(self):
        self.sandbox = Sandbox()
        self.obj = Object3D()

    def tearDown(self):
        self.sandbox.restore()

    def test_position(self):
        obj = self.obj
        obj.pos.x = 10
        self.assertEqual(obj._position[0], 10)
        obj.position.y = 8
        self.assertEqual(obj._position[1], 8)
        obj.pos.z = 3
        self.assertEqual(obj._position[2], 3)

    def test_add_objects(self):
        obj = self.obj
        self.sandbox.stub(obj, '_add_child')

        obj.add(DummyObject(), DummyObject(), DummyObject())
        self.assertEqual(obj._add_child.call_count, 3)

    def test_add_child(self):
        obj = self.obj
        child = DummyObject()
        obj._add_child(child)
        self.assertEqual(child.parent, obj)
        self.assertEqual(len(obj.children), 1)


if __name__ == "__main__":
    unittest.main()
