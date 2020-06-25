"""Simple example of setting up a multi-agent policy mapping.

Control the number of agents and policies via --num-agents and --num-policies.

This works with hundreds of agents and policies, but note that initializing
many TF policies will take some time.

Also, TF evals might slow down with large numbers of policies. To debug TF
execution, set the TF_TIMELINE_DIR environment variable.
"""

from pdb import set_trace as T

import argparse
import gym
import random

from gym import spaces

import ray
from ray import tune
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
from ray.rllib.examples.models.shared_weights_model import \
    SharedWeightsModel1, SharedWeightsModel2, TorchSharedWeightsModel
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from ray.rllib.models.torch.torch_action_dist import TorchMultiActionDistribution, TorchCategorical, TorchDiagGaussian

from ray.rllib.utils import try_import_tree

import torch
from torch import nn

tf = try_import_tf()

parser = argparse.ArgumentParser()

parser.add_argument("--num-agents", type=int, default=4)
parser.add_argument("--num-policies", type=int, default=2)
parser.add_argument("--stop-iters", type=int, default=20)
parser.add_argument("--stop-reward", type=float, default=150)
parser.add_argument("--stop-timesteps", type=int, default=100000)
parser.add_argument("--simple", action="store_true")
parser.add_argument("--num-cpus", type=int, default=1)
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--torch", action="store_true")

class Env(MultiAgentCartPole):
   def step(self, action_dict):
      return super().step(action_dict['Obs'])

class Policy(TorchModelV2, nn.Module):
   def __init__(self, *args, **kwargs):
      TorchModelV2.__init__(self, *args, **kwargs)
      nn.Module.__init__(self)

      self.net = nn.Linear(4, 4)
      self.vf  = nn.Linear(4, 4)

   def forward(self, input_dict, state, seq_lens):
      obs = input_dict['obs_flat']
      logits = self.net(obs)
      self.value = self.vf(obs)

      distrib = TorchMultiActionDistribution(
         logits,
         model=self,
         child_distributions=[TorchCategorical],
         input_lens=[4],
         action_space=spaces.Dict({'Obs': spaces.Discrete(4)}))

      atns = distrib.sample()
      tree = try_import_tree()
      flat = tree.flatten(atns)
      return logits, []

   def value_function(self):
      return self.value

if __name__ == "__main__":
    args = parser.parse_args()

    ray.init(num_cpus=args.num_cpus or None)

    # Register the models to use.
    ModelCatalog.register_custom_model("model1", Policy)
    ModelCatalog.register_custom_model("model2", Policy)

    # Get obs- and action Spaces.
    single_env = gym.make("CartPole-v0")
    obs_space = single_env.observation_space
    act_space = single_env.action_space
    act_space = spaces.Dict({'Obs': spaces.Discrete(4)})

    # Each policy can have a different configuration (including custom model).
    def gen_policy(i):
        config = {
            "model": {
                "custom_model": ["model1", "model2"][i % 2],
            },
            "gamma": random.choice([0.95, 0.99]),
        }
        return (None, obs_space, act_space, config)

    # Setup PPO with an ensemble of `num_policies` different policies.
    policies = {
        "policy_{}".format(i): gen_policy(i)
        for i in range(args.num_policies)
    }
    policy_ids = list(policies.keys())

    config = {
        "env": Env,
        "env_config": {
            "num_agents": args.num_agents,
        },
        "log_level": "DEBUG",
        "simple_optimizer": args.simple,
        "num_sgd_iter": 10,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": (lambda agent_id: random.choice(policy_ids)),
        },
        "use_pytorch": args.torch,
    }
    stop = {
        "episode_reward_mean": args.stop_reward,
        "timesteps_total": args.stop_timesteps,
        "training_iteration": args.stop_iters,
    }

    import ray.rllib.agents.ppo.ppo as ppo
    trainer = ppo.PPOTrainer(config=config)

    env = Env({'num_agents': args.num_agents})
    obs = env.reset()
    atn = trainer.compute_action(obs[0], policy_id='policy_0')
