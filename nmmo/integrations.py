from pdb import set_trace as T

from nmmo import Env

try:
    import supersuit as ss
except ImportError:
    raise ImportError('Integrations depend on supersuit. Install and then retry')


def rllib_env_cls():
    try:
        from ray import rllib
    except ImportError:
        raise ImportError('Integrations depend on rllib. Install ray[rllib] and then retry')

    import rllib
    class RLlibEnv(Env, rllib.MultiAgentEnv):
        def __init__(self, config):
            self.config = config['config']
            self.config.EMULATE_CONST_HORIZON = True
            super.__init__(self.config)

        def render(self):
            #Patrch of RLlib dupe rendering bug
            if not self.config.RENDER:
                return

            super().render()

        def step(self, actions):
            obs, rewards, dones, infos = super().step(actions)

            population  = len(self.realm.players) == 0
            hit_horizon = self.realm.tick >= self.config.EMULATE_CONST_HORIZON

            dones['__all__'] = False
            if not self.config.RENDER and (hit_horizon or population):
                dones['__all__'] = True

            return obs, rewards, dones, infos
        

class SB3Env(Env):
    def __init__(self, config):
        config.EMULATE_FLAT_OBS      = True
        config.EMULATE_FLAT_ATN      = True
        config.EMULATE_CONST_NENT    = True
        config.EMULATE_CONST_HORIZON = True

        super().__init__(config)

    def step(self, actions):
        assert type(actions) == dict

        obs, rewards, dones, infos = super().step(actions)

        if self.realm.tick >= self.config.HORIZON or len(self.realm.players) == 0:
            # Cheat logs into infos
            infos[1]['logs'] = self.terminal()['Stats']

        return obs, rewards, dones, infos 

class CleanRLEnv(SB3Env):
    def __init__(self, config):
        super().__init__(config)

def sb3_vec_envs(config_cls, num_envs, num_cpus):
    config = config_cls()
    env    = SB3Env(config)

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env.black_death = True #We provide our own black_death emulation
    env = ss.concat_vec_envs_v1(env, num_envs, num_cpus,
            base_class='stable_baselines3')

    return env

def cleanrl_vec_envs(config_cls, num_envs, num_cpus):
    config = config_cls()
    env    = CleanRLEnv(config)

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env.black_death = True #We provide our own black_death emulation

    env = ss.concat_vec_envs_v1(env, num_envs, num_cpus,
            base_class='gym')

    env.single_observation_space = env.observation_space
    env.single_action_space      = env.action_space
    env.is_vector_env            = True

    return env
