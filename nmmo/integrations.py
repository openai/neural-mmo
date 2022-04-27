from pdb import set_trace as T

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

def cleanrl_vec_envs(config_cls, eval_config_cls=None, verbose=True):
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
                    config.NUM_ENVS // config.NENT,
                    config.NUM_CPUS,
                    base_class='gym')

            env.single_observation_space = env.observation_space
            env.single_action_space      = env.action_space
            env.is_vector_env            = True

            return env
        return make_env

    # Sanity check config cls
    assert isinstance(config_cls, type), 'config_cls must be a type (did ytou pass an instance?)'
    assert hasattr(config_cls, 'NUM_ENVS'), 'config_cls must define NUM_ENVS'
    assert hasattr(config_cls, 'NUM_CPUS'), 'config_cls must define NUM_CPUS'


    envs      = make_env_fn(config_cls)

    config    = config_cls()
    dummy_env = CleanRLEnv(config)
    if eval_config_cls is not None:
        assert hasattr(config_cls, 'NUM_ENVS'), 'eval_config_cls must define NUM_ENVS'
        assert hasattr(eval_config_cls, 'NUM_CPUS'), 'eval_config_cls must define NUM_CPUS'
        assert isinstance(eval_config_cls, type), 'eval_config_cls must be a type (did ytou pass an instance?)'
        envs        = [make_env_fn(config_cls), make_env_fn(eval_config_cls)]

        eval_config = eval_config_cls()
        num_cpus    = config.NUM_CPUS + eval_config.NUM_CPUS
        num_envs    = config.NUM_ENVS // config.NENT + eval_config.NUM_ENVS // eval_config.NENT
    else:
        envs     = [make_env_fn(config_cls)]

        num_cpus = config.NUM_CPUS
        num_envs = config.NUM_ENVS // config.NENT

    envs = ss.vector.MakeCPUAsyncConstructor(num_cpus)(envs,
            obs_space=dummy_env.observation_space(1),
            act_space=dummy_env.action_space(1))
    envs.is_vector_env = True

    if verbose:
        print(f'Created {num_envs} envs across {num_cpus} cores')

    return envs
