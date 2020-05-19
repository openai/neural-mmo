from pdb import set_trace as T
import numpy as np

class Repeated():
   def __init__(self, x):
      self.data = x
      self.cls  = type(x)
  
   def __repr__(self):
      return str(self.__class__) + str(self.data)

raw_data = {
   'Obs_0': [
      {
         'Attribute_0': 1,
         'Attribute_1': 2,
         'Attribute_2': 3,
         'Attribute_3': 4,
      }
   ],
   'Obs_1': [
      {
         'Attribute_0': 5,
         'Attribute_1': 6,
         'Attribute_2': 7,
         'Attribute_3': 8,

      },
      {
         'Attribute_0': 9,
         'Attribute_1': 10,
         'Attribute_2': 11,
         'Attribute_3': 12,
      }
   ]
}

outer_repeat = Repeated({
   'Obs_0': Repeated([
      Repeated({
         'Attribute_0': 1,
         'Attribute_1': 2,
         'Attribute_2': 3,
         'Attribute_3': 4,
      })
   ]),
   'Obs_1': Repeated([
      Repeated({
         'Attribute_0': 5,
         'Attribute_1': 6,
         'Attribute_2': 7,
         'Attribute_3': 8,

      }),
      Repeated({
         'Attribute_0': 9,
         'Attribute_1': 10,
         'Attribute_2': 11,
         'Attribute_3': 12,
      })
   ])
})

inner_repeat = {
   'Obs_0': {
      Repeated({
         'Attribute_0': 1,
         'Attribute_1': 2,
         'Attribute_2': 3,
         'Attribute_3': 4,
      })
   },
   'Obs_1': {
      Repeated({
         'Attribute_0': 5,
         'Attribute_1': 6,
         'Attribute_2': 7,
         'Attribute_3': 8,

      }),
      Repeated({
         'Attribute_0': 9,
         'Attribute_1': 10,
         'Attribute_2': 11,
         'Attribute_3': 12,
      })
   }
}

def batchUnified(batch, shape):
   if type(batch) == Repeated:
      struct = np.zeros(shape, dtype=np.ndarray)
      if batch.cls in (list, set):
         iterable = batch.data
      elif batch.cls == dict:
         iterable = batch.data.values()
      for obs, observation in enumerate(iterable):
         struct[obs] = batchUnified(observation, shape[1:])
   elif type(batch) == dict:
      struct = {}
      for obs, observation in batch.items():
         struct[obs] = batchUnified(observation, shape[1:])
   elif type(batch) in (list, set):
      struct = []
      for observation in batch:
         struct.append(batchUnified(observation, shape[1:]))      
   else: #Primitive
      return batch
   return struct

if __name__ == '__main__':
   print('Raw Data:\n', raw_data, '\n')
   print('Outer Repeat:\n', batchUnified(outer_repeat, (2, 3, 4)), '\n')
   print('Inner Repeat:\n', batchUnified(inner_repeat, (2, 3, 4)))
   

