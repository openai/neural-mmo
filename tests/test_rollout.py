from pdb import set_trace as T
import nmmo


def test_rollout():
   config = nmmo.config.Default()  
   config.AGENTS = [nmmo.core.agent.Random]

   env = nmmo.Env()
   env.reset()
   for i in range(128):
       env.step({})

if __name__ == '__main__':
   test_rollout()
