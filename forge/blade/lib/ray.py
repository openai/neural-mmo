from pdb import set_trace as T
import ray, os
import time

OBJECT_INFO_PREFIX = b"OI:"
OBJECT_LOCATION_PREFIX = b"OL:"
TASK_TABLE_PREFIX = b"TT:"

def init(config, mode):
   os.environ['MKL_NUM_THREADS'] = '1'
   os.environ['OMP_NUM_THREADS'] = '1'

   if mode == 'local':
      ray.init(local_mode=True)
   elif mode == 'default':
      ray.init()
   elif mode == 'remote':
      ray.init(redis_address=config.HOST + ':6379')
      print('Cluster started with resources:')
      print(ray.cluster_resources())
   
   else:
      print('Invalid ray mode (local/default/remote)')
      exit(-1)

def clearbuffers():
   start = time.time()
   print('Clearing ray buffers...')
   flush_task_and_object_metadata_unsafe()
   flush_redis_unsafe()
   print('Done. Free time: ', time.time() - start)

#Continuous moving average
class CMA():
   def __init__(self):
      self.t = 1.0
      self.cma = None

   def update(self, x):
      if self.cma is None:
         self.cma = x
         return
      self.cma = (x + self.t*self.cma)/(self.t+1)
      self.t += 1.0

#Continuous moving average
class CMV():
   def __init__(self):
      self.cma = CMA()
      self.cmv = None

   def update(self, x):
      if self.cmv is None:
         self.cma.update(x)
         self.cmv = 0
         return
      prevMean = self.cma.cma
      self.cma.update(x)
      self.cmv += (x-prevMean)*(x-self.cma.cma)

   @property
   def stats(self):
      return self.cma.cma, self.cmv

class RayBenchmark:
   def __init__(self):
      self.cmv = CMV()
   def startRecord(self):
      self.start = time.time()

   def stopRecord(self):
      delta = time.time() - self.start
      self.cmv.update(delta)

   def reset(self):
      self.cmv = CMV()
      
   @property
   def stats(self):
      mean, var = self.cmv.stats
      return {'mean': mean, 'var': var}

def put(*args, profile=None):
   if profile is not None:
      if not hasattr(put, 'bench'):
         put.bench = {}
      if not profile in put.bench:
         put.bench[profile] = RayBenchmark()
      put.bench[profile].startRecord()
   ret = ray.put(*args)
   if profile is not None:
      put.bench[profile].stopRecord()
   return ret

def get(*args, profile=None):
   if profile is not None:
      if not hasattr(get, 'bench'):
         get.bench = {}
      if not profile in get.bench:
         get.bench[profile] = RayBenchmark()
      get.bench[profile].startRecord()
   ret = ray.get(*args)
   if profile is not None:
      get.bench[profile].stopRecord()
   return ret

def profile():
   stats = {}
   for k, f in zip('put get'.split(), (put, get)):
      if hasattr(f, 'bench'):
         stats[k] = dict((k, v.stats) for k, v in put.bench.items())
   return stats

def flush_redis_unsafe():
   """This removes some non-critical state from the primary Redis shard.

   This removes the log files as well as the event log from Redis. This can
   be used to try to address out-of-memory errors caused by the accumulation
   of metadata in Redis. However, it will only partially address the issue as
   much of the data is in the task table (and object table), which are not
   flushed.
   """
   if not hasattr(ray.worker.global_worker, "redis_client"):
       raise Exception("ray.experimental.flush_redis_unsafe cannot be called "
                       "before ray.init() has been called.")

   redis_client = ray.worker.global_worker.redis_client

   # Delete the log files from the primary Redis shard.
   keys = redis_client.keys("LOGFILE:*")
   if len(keys) > 0:
       num_deleted = redis_client.delete(*keys)
   else:
       num_deleted = 0
   print("Deleted {} log files from Redis.".format(num_deleted))

   # Delete the event log from the primary Redis shard.
   keys = redis_client.keys("event_log:*")
   if len(keys) > 0:
       num_deleted = redis_client.delete(*keys)
   else:
       num_deleted = 0
   print("Deleted {} event logs from Redis.".format(num_deleted))


def flush_task_and_object_metadata_unsafe():
   """This removes some critical state from the Redis shards.

   This removes all of the object and task metadata. This can be used to try
   to address out-of-memory errors caused by the accumulation of metadata in
   Redis. However, after running this command, fault tolerance will most
   likely not work.
   """
   if not hasattr(ray.worker.global_worker, "redis_client"):
       raise Exception("ray.experimental.flush_redis_unsafe cannot be called "
                       "before ray.init() has been called.")

   def flush_shard(redis_client):
       num_task_keys_deleted = 0
       for key in redis_client.scan_iter(match=TASK_TABLE_PREFIX + b"*"):
           num_task_keys_deleted += redis_client.delete(key)
       print("Deleted {} task keys from Redis.".format(num_task_keys_deleted))

       num_object_keys_deleted = 0
       for key in redis_client.scan_iter(match=OBJECT_INFO_PREFIX + b"*"):
           num_object_keys_deleted += redis_client.delete(key)
       print("Deleted {} object info keys from Redis.".format(
                 num_object_keys_deleted))

       num_object_location_keys_deleted = 0
       for key in redis_client.scan_iter(match=OBJECT_LOCATION_PREFIX + b"*"):
           num_object_location_keys_deleted += redis_client.delete(key)
       print("Deleted {} object location keys from Redis.".format(
                 num_object_location_keys_deleted))

   for redis_client in ray.worker.global_state.redis_clients:
       flush_shard(redis_client)
