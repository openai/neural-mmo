from pdb import set_trace as T
import numpy as np
import ray
import time

ray.init('localhost:6379')

@ray.remote
def f():
    time.sleep(0.01)
    return ray.services.get_node_ip_address()

# Get a list of the IP addresses of the nodes that have joined the cluster.
print(set(ray.get([f.remote() for _ in range(100)])))
print(ray.cluster_resources())
