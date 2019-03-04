from pdb import set_trace as T
import tensorflow as tf
import numpy as np
import time


class Logger:
   def __init__(self):
      self.writer = tf.summary.FileWriter("./logdir")

   def log(self, val, tag):
      summary = tf.Summary(value=[tf.Summary.Value(
            tag=tag, simple_value=val)])
      self.writer.add_summary(summary)

i = 0
logger = Logger()
while True:
   i += 1
   s = np.sin(i/100)
   logger.log(s, 'sin')
   logger.log(np.log(i), 'log')

   if i % 30 == 0:
      print(s)

   time.sleep(0.01)
    
# while the above is running, execute
#     tensorboard --logdir ./logdir

