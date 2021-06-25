def waittime(func):
   '''Ascend performance profiling decorator. Apply to methods of Ascend
   subclasses that are purely waiting for remotes to synchronize'''
   pass

def runtime(func):
   '''Ascend performance profiling decorator. Apply to top-level methods of
   Ascend subclasses that performing expensive computations within the
   current infrastructure (hardware) layer.'''
   pass

class Ascend:
   '''This module is a stub reference to the Ascend core that documents
      the external API. Ascend is a featherweight Ray wrapper for persistent,
      multilevel, synchronous + asynchronous computation that works with pdb
      (local mode). Internal docs available at :mod:`forge.trinity.ascend`'''
   def __init__(self, disciple, n, *args):
      '''
      Args:
         disciple: A class to instantiate remotely
         n       : The number of disciples to instantiate
         *args   : Arbitrary arguments for all disciples
      '''
      pass

   def distribute(self, *args, shard=None):
      '''Asynchronously invoke the step method of all remote disciples

      Args:
         *args : List of arguments to broadcast to all disciples. Shared
            arguments may have any type. Sharded arguments must be wrapped
            in a dictionary specifying target client shards::

               {
                  0: data_for_shard_0,
                  1: data_for_shard_1,
                  ...
               }

         shard : A boolean mask (list) of the same length as *args*
            specifying which arguments to shard::
               
               True: shard the current argument across remote clients
               False: share the current argument across remote clients
         
      Returns:
         list:

         rets:
            A list of async step return handles from remote disciples
      '''
      pass

   def synchronize(self, rets):
      '''Synchronizes asynchronous returns from *distribute*

      Args:
         rets: list of async handles returned from distribute

      Returns:
         list:

         rets:
            A list of returns from all disciples
      '''
      pass

   def step(self, *args, shard=False):
      '''Synchronously invoke the step method of all remote disciples

      Args:
         *args : Arguments to broadcast to all disciples. See *distribute*
         shard : A boolean mask. See *distribute*
 
      Returns:
         list:

         rets:
            A list of returns from all disciples
      '''
      pass

   def discipleLogs(self):
      '''Logging objects from all disciples

      Returns:
         list:

         logs:
            A list of disciple logs
      '''
      pass
 
   def localize(func, remote):
      '''Enable remote functions to be called without explicitly specifying
      func.remote

      Args:
         func   : Function to localize
         remote : True if *func* is remote, False otherwise
 
      Returns:
         boolean:

         f:
            A localized function (f or f.remote)
      '''
      pass

   def isRemote(obj):
      '''Check if an object is a remote Ray instance

      Returns:
         boolean:

         isRemote:
            True if *obj* is remote, False otherwise'''
      pass
