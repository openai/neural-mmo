import heapq, itertools
import itertools

class PriorityQueue:
   def __init__(self, capacity, unique=False):
      self.q, self.items = [], set()
      self.capacity = capacity
      self.count = itertools.count()
      self.unique = unique

   def get(self, ind):
      priority, item = self.tolist()[ind]
      return priority, item

   def push(self, item, priority, uniqueKey=None):
      if self.unique:
         self.items.add(uniqueKey)
      count = next(self.count)
      if len(self.q) >= self.capacity:
         return heapq.heappushpop(self.q, (priority, count, item))
      heapq.heappush(self.q, (priority, count, item))

   def pop(self):
      priority, _, item = heapq.heappop(self.q)
      if self.unique:
         self.items.remove(item)
      return priority, item

   @property
   def peek(self):
      return self.peekPriority, self.peekValue

   @property
   def peekPriority(self):
      ret = heapq.nlargest(1, self.q)
      if len(ret) > 0:
         return ret[0][0]

   @property
   def peekValue(self):
      ret = heapq.nlargest(1, self.q)
      if len(ret) > 0:
         return ret[0][2]


   def tolist(self):
      q = heapq.nlargest(self.n, self.q)
      return [(e[0], e[2]) for e in q]

   def priorities(self):
      return sorted([e[0] for e in self.q], reverse=True)

   def print(self):
      q = heapq.nlargest(self.n, self.q)
      print([(e[0]) for e in q], end='')
      print()

   @property
   def n(self):
      return len(self.q)

