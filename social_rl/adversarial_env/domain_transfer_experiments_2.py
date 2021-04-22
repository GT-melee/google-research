"""
1. load env
2. run trained agent in random env 1000 times vs trained agent without domain shift
3. Compare % success
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import tensorflow as tf
from tf_agents.policies import policy_loader
# from tf_agents.google.utils import mp4_video_recorder
from tf_agents.trajectories import time_step as ts_lib

import gym
from social_rl import gym_multigrid

tf.compat.v1.enable_v2_behavior()
from social_rl.adversarial_env import adversarial_env

# TODO
# source_shift = []
# target_shift = []


# all_source_agents = ["/home/charlie/SDRIVE/datasets/no_shift/policy_saved_model/agent/0/policy_000499950"]
# all_source_agents = ["/home/ddobre/Projects/mila/game_theory/dynamic_shift/policy_000499950"]
# all_source_agents = ["/home/ddobre/Projects/mila/game_theory/dynamic_shift/new_runs/policy_000038700"]
all_source_agents = [
    "/home/ddobre/Projects/mila/game_theory/dynamic_shift/new_runs/policy_000329400"
]

all_variable_agents = []

all_source_agents = [tf.compat.v2.saved_model.load(a) for a in all_source_agents]
all_variable_agents = [tf.compat.v2.saved_model.load(a) for a in all_variable_agents]


colors = [5, 1, 6]  # wall, goal, floor: gray, green, black
sequences = [[12, 0, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]]  # corridor


# class Experiment:
#     def __init__(self, 
#                  name,
#                  adv_env_name='MultiGrid-Adversarial-v0',
#                  seeds: int = 0,
#                  root_dir: str = None,
#                  fps: int = 4):
#         self.name = name
#         self.adv_env_name = adv_env_name
#         self.seeds = seeds
#         self.fps = fps

#         if root_dir is None:
#             root_dir = "/tmp/adversarial_env/"
#         self.root_dir = Path(root_dir)
#         self.videos_dir = self.root_dir / "videos"

#         # TODO: models dir

#     def load_agents(self):
#         pass

#     def run_agent(self,
#                   policy,
#                   recorder,
#                   env_name,
#                   py_env,
#                   tf_env,
#                   encoded_images=None):

#         pass


def get_envs():
    names = []
    targets = []
    sources = []

    for i, sequence in enumerate(sequences):
        # for shift in [source_shift, target_shift]:
        py_env = adversarial_env.load("MultiGrid-Adversarial-v0")
        tf_env = adversarial_env.AdversarialTFPyEnvironment(py_env)

        tf_env.reset()

        done = False
        if colors is not None:
            for c in colors:
                _, _, done, _ = tf_env.step_adversary(c)

        for s in sequence:
            _, _, done, _ = tf_env.step_adversary(s)

        #    plt.imshow(tf_env.render('rgb_array'))
        # plt.show()
#        while not done:
#            _, _, done, _ = tf_env.step_adversary(s)

        plt.imshow(py_env.render())
        plt.show(block=False)

        # todo apply shift
        # targets.append(TODO)
        # sources.append(TODO)
        targets.append(tf_env)
        sources.append(tf_env)
        names.append(i)

    return zip(sources, targets, names)


def load_agent(checkpoint_path):
    agent = tf.compat.v2.saved_model.load(checkpoint_path)
    return agent


def backwards_compatible_timestep(new_ts):
    """Remove new observations added in later versions of the environment."""
    old_obs = {"image": new_ts.observation["image"], "time_step": new_ts.observation["time_step"]}
    return ts_lib.TimeStep(new_ts.step_type, new_ts.reward, new_ts.discount, old_obs)


def run_agent_on_env(agent, env):
    done = False
    rewards = []

    import ipdb; ipdb.set_trace()

    # make the environment
    # env.reset()
    # env.step_adversary(adversarial_steps[0])
    # env.step_adversary(adversarial_steps[1])
    # for i in adversarial_steps[2:]:
        # env.step_adversary(i)
    # reset_img = env.render('rgb_array')
   
    timestep = env.reset_agent()
    policy_state = agent.get_initial_state(1)

    plt.imshow(env.render()[0])
    # plt.show()
 
    actions = []

    #   while not done:
    for i in range(100):
        policy_step = agent.action(timestep, policy_state=policy_state)

        policy_state = policy_step.state
        actions.append(policy_step.action.numpy()[0])

        timestep = env.step(policy_step.action)
        # rewards.append(reward)
    print(actions)
    plt.imshow(env.render()[0])
    plt.show()
 
    return np.array(rewards)


by_env = {}
for i, seq in enumerate(sequences):
    by_env[i] = {}
    for j, agent in enumerate(all_source_agents + all_variable_agents):
        by_env[i] = {}
        by_env[i]["source"] = {}
        by_env[i]["target"] = {}


def eval_agent(agent, agent_name):
    global by_env

    # TODO: fix

    for source, target, env_name in get_envs():
        source_rews = run_agent_on_env(agent, source)
        target_rews = run_agent_on_env(agent, target)

        by_env[env_name][agent_name]["source"] = source_rews
        by_env[env_name][agent_name]["target"] = target_rews


for i, agent in enumerate(all_source_agents + all_variable_agents):
    eval_agent(agent, i)
