from pdb import set_trace as T

import nmmo
from nmmo import Env

def rllib_env_cls():
    try:
        from ray import rllib
    except ImportError:
        raise ImportError('Integrations depend on rllib. Install ray[rllib] and then retry')
    class RLlibEnv(Env, rllib.MultiAgentEnv):
        def __init__(self, config):
            self.config = config['config']
            self.config.EMULATE_CONST_HORIZON = True
            super().__init__(self.config)

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

    return RLlibEnv

class SB3Env(Env):
    def __init__(self, config):
        config.EMULATE_FLAT_OBS      = True
        config.EMULATE_FLAT_ATN      = True
        config.EMULATE_CONST_PLAYER_N = True
        config.EMULATE_CONST_HORIZON = True

        super().__init__(config)

    def step(self, actions):
        assert type(actions) == dict

        obs, rewards, dones, infos = super().step(actions)

        if self.realm.tick >= self.config.HORIZON or len(self.realm.players) == 0:
            # Cheat logs into infos
            stats = self.terminal()
            stats = {**stats['Env'], **stats['Player'], **stats['Milestone'], **stats['Event']}

            infos[1]['logs'] = stats

        return obs, rewards, dones, infos 

class CleanRLEnv(SB3Env):
    def __init__(self, config):
        super().__init__(config)

def sb3_vec_envs(config_cls, num_envs, num_cpus):
    try:
        import supersuit as ss
    except ImportError:
        raise ImportError('SB3 integration depend on supersuit. Install and then retry')

    config = config_cls()
    env    = SB3Env(config)

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env.black_death = True #We provide our own black_death emulation
    env = ss.concat_vec_envs_v1(env, num_envs, num_cpus,
            base_class='stable_baselines3')

    return env

def cleanrl_vec_envs(config_classes, verbose=True):
    '''Creates a vector environment object from a list of configs.

    Each subenv points to a single agent, but many agents can share the same env.
    All envs must have the same observation and action space, but they can have
    different numbers of agents'''

    try:
        import supersuit as ss
    except ImportError:
        raise ImportError('CleanRL integration depend on supersuit. Install and then retry')

    def make_env_fn(config_cls):
        '''Wraps the make_env fn to add a a config argument'''
        def make_env():
            config = config_cls()
            env    = CleanRLEnv(config)

            env = ss.pettingzoo_env_to_vec_env_v1(env)
            env.black_death = True #We provide our own black_death emulation

            env = ss.concat_vec_envs_v1(env,
                    config.NUM_ENVS // config.PLAYER_N,
                    config.NUM_CPUS,
                    base_class='gym')

            env.single_observation_space = env.observation_space
            env.single_action_space      = env.action_space
            env.is_vector_env            = True

            return env
        return make_env

    dummy_env = None
    all_envs = [] 

    num_cpus   = 0
    num_envs   = 0
    num_agents = 0

    if type(config_classes) != list:
        config_classes = [config_classes]

    for idx, cls in enumerate(config_classes):
        assert isinstance(cls, type), 'config_cls must be a type (did ytou pass an instance?)'
        assert hasattr(cls, 'NUM_ENVS'), f'config class {cls} must define NUM_ENVS'
        assert hasattr(cls, 'NUM_CPUS'), f'config class {cls} must define NUM_CPUS'
        assert isinstance(cls, type), f'config class {cls} must be a type (did you pass an instance?)'

        if dummy_env is None:
            config    = cls()
            dummy_env = CleanRLEnv(config)

        #neural = [e == nmmo.Agent for e in cls.PLAYERS]
        #n_neural = sum(neural) / len(neural) * config.NUM_ENVS
        #assert int(n_neural) == n_neural, f'{sum(neural)} neural agents and {cls.PLAYER_N} classes'
        #n_neural = int(n_neural)
        
        envs = make_env_fn(cls)#, n_neural)
        all_envs.append(envs)

        # TODO: Find a cleaner way to specify env scale that enables multiple envs per CPU
        # without having to pass multiple configs
        num_cpus    += cls.NUM_CPUS
        num_envs    += cls.NUM_CPUS
        num_agents  += cls.NUM_CPUS * cls.PLAYER_N



    envs = ss.vector.ProcConcatVec(all_envs,
            dummy_env.observation_space(1),
            dummy_env.action_space(1),
            num_agents,
            dummy_env.metadata)
    envs.is_vector_env = True

    if verbose:
        print(f'nmmo.integrations.cleanrl_vec_envs created {num_envs} envs across {num_cpus} cores')

    return envs
