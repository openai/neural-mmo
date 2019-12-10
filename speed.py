from pdb import set_trace as T
import time

class Foo:
   def __init__(self):
      bar = 0


n = 10000
foo = Foo()

start = time.time()
for i in range(n):
   foo.bar = 1
print(time.time() - start)

start = time.time()
for i in range(n):
   foo.__dict__['bar'] = 1
print(time.time() - start)

start = time.time()
for i in range(n):
   setattr(foo, 'bar', 1)
print(time.time() - start)

start = time.time()
for i in range(n):
   foo.__setattr__('bar', 1)
print(time.time() - start)

start = time.time()
for i in range(n):
   bar = foo.bar
print(time.time() - start)

start = time.time()
for i in range(n):
   bar = foo.__dict__['bar']
print(time.time() - start)

start = time.time()
for i in range(n):
   bar = getattr(foo, 'bar')
print(time.time() - start)

start = time.time()
for i in range(n):
   bar = foo.__getattribute__('bar')
print(time.time() - start)


