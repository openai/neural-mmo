from pdb import set_trace as T


def test_import():
   import nmmo

def test_env_creation():
   import nmmo
   env = nmmo.Env()
   env.reset()
   env.step({})

def test_io():
   import nmmo
   env = nmmo.Env()
   env.observation_space(0)
   env.action_space(0)

if __name__ == '__main__':
   test_io()
