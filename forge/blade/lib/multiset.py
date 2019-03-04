from collections import defaultdict

class MultiSet:
   def __init__(self, capacity=0):
      self.data = defaultdict(int)
      self.count = 0
      self.capacity = capacity

   def __iter__(self):
      return self.data.__iter__()

   def countItem(self, item):
      return self.data[item]

   @property
   def full(self):
      return self.capacity != 0 and self.count >= self.capacity

   @property
   def empty(self):
      return self.count == 0

   def get(self, item):
      return self.data[item]

   def isIn(self, item, num=1):
      return self.data[item] > num

   def add(self, item, num=1):
      assert self.capacity == 0 or self.count+num <= self.capacity
      self.data[item] += num
      self.count += num
   
   #Alias
   def remove(self, item, num=1):
      self.pop(item, num)

   def pop(self, item, num=1):
      assert self.capacity == 0 or self.count-num >= 0
      self.data[item] -= num
      self.count -= num

   def union(self, other):
      for e in other:
         self.add(e, other.count(e))

   def diff(self, other):
      for e in other:
         self.remove(e, other.count(e))

   def contains(self, other):
      for e in other:
         if not isIn(e, other[e]):
            return False
      return True

