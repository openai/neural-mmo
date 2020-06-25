import ray
import time

@ray.remote
def work():
    time.sleep(3)

if __name__ == '__main__':
    ray.init()
    t = time.time()
    rets = [work.remote() for _ in range(96)]
    ray.get(rets)
    print('Time: {}'.format(time.time() - t))

