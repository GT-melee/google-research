"""
1. load env
2. run trained agent in random env 1000 times vs trained agent without domain shift
3. Compare % success
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# tensorflow
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.policies import policy_loader
from tf_agents.trajectories import time_step as ts_lib

# gym stuff
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from tf_agents.typing.types import PyEnv

# multigrid
from social_rl import gym_multigrid
from social_rl.adversarial_env import adversarial_env
from social_rl.multiagent_tfagents import multiagent_gym_suite
from social_rl.gym_multigrid.gym_minigrid.minigrid import COLORS, IDX_TO_COLOR, COLOR_TO_IDX

tf.compat.v1.enable_v2_behavior()
logging.basicConfig(level=logging.INFO)


VAL_ENVS = [
    'MultiGrid-TwoRooms-Minigrid-v0',
    'MultiGrid-Cluttered40-Minigrid-v0',
    'MultiGrid-Cluttered10-Minigrid-v0',
    'MultiGrid-SixteenRooms-v0',
    'MultiGrid-Maze2-v0',
    'MultiGrid-Maze3-v0',
    'MultiGrid-Labyrinth2-v0',
]
TEST_ENVS = [
    'MultiGrid-FourRooms-Minigrid-v0',
    'MultiGrid-Cluttered50-Minigrid-v0',
    'MultiGrid-Cluttered5-Minigrid-v0',
    'MultiGrid-Empty-Random-15x15-Minigrid-v0',
    'MultiGrid-SixteenRoomsFewerDoors-v0',
    'MultiGrid-Maze-v0',
    'MultiGrid-Labyrinth-v0',
]
MINI_VAL_ENVS = [
    'MultiGrid-MiniTwoRooms-Minigrid-v0',
    'MultiGrid-Empty-Random-6x6-Minigrid-v0',
    'MultiGrid-MiniCluttered6-Minigrid-v0',
    'MultiGrid-MiniCluttered-Lava-Minigrid-v0',
    'MultiGrid-MiniMaze-v0'
]
MINI_TEST_ENVS = [
    'MultiGrid-MiniFourRooms-Minigrid-v0',
    'MultiGrid-MiniCluttered7-Minigrid-v0',
    'MultiGrid-MiniCluttered1-Minigrid-v0'
]


# Define the Environment loaders for base and adversarial environments
class BaseEnv:
    def __init__(self, name: str, colors: str = None, video_fp: str = None, gym_kwargs=None):
        self.name = name
        self.video_fp = video_fp

        if gym_kwargs is None:
            gym_kwargs = {}
        self.gym_kwargs = gym_kwargs
        self.colors = colors

        self.py_env = None
        self.tf_env = None

    def make_env(self):

        kwargs = self.gym_kwargs.copy()
        if self.colors is not None:
            kwargs.update({
                "wall_color": IDX_TO_COLOR[self.colors[0]] ,
                "goal_color": IDX_TO_COLOR[self.colors[1]] ,
                "floor_color": IDX_TO_COLOR[self.colors[2]] ,
            })

        py_env = multiagent_gym_suite.load(self.name, gym_kwargs=kwargs)
        tf_env = tf_py_environment.TFPyEnvironment(py_env)
        py_env.reset()
        tf_env.reset()
        if self.video_fp is not None:
            py_env = VideoRecorder(py_env, self.video_fp)
            py_env.capture_frame()

        self.py_env, self.tf_env = py_env, tf_env
        return py_env, tf_env

    def close(self):
        if self.py_env is not None:
            self.py_env.close()


class AdvEnv(BaseEnv):
    def __init__(self, sequence: list = None, adversary_agent = None, colors: list = None, video_fp: str = None):
        """
        Args:
            sequence: list of locs where to place the agent, goal, and then walls
            colors (optional): specifies the (wall, goal, floor) color
        """
        # if colors is None:
        #     # default color scheme
        #     # colors = [5, 1, 6]
        #     pass
        
        # self.colors = colors

        gym_kwargs = {}
        if colors is not None:
            gym_kwargs["domain_shifts"] = True

        super().__init__(name='MultiGrid-Adversarial-v0', colors=colors, video_fp=video_fp, gym_kwargs=gym_kwargs)
        
        self.py_env = None
        self.tf_env = None

        # TODO: figure out if using sequence or agent
        if sequence is not None and adversary_agent is not None:
            raise TypeError
        
        self.agent = None
        self.sequence = None
        self.agent = None

        if sequence is not None:
            self.mode = "sequence"
            self.sequence = sequence

        elif adversary_agent is not None:
            self.mode = "agent"
            self.agent = adversary_agent
        
        else:
            raise NotImplementedError

    def make_env(self):
        """Also starts video recorder"""
        py_env = adversarial_env.load(self.name, gym_kwargs=self.gym_kwargs)
        tf_env = adversarial_env.AdversarialTFPyEnvironment(py_env)

        py_env.reset()
        tf_env.reset()
        if self.video_fp is not None:
            py_env = VideoRecorder(py_env, self.video_fp)

        self.py_env, self.tf_env = py_env, tf_env
        return py_env, tf_env


    def adversarial_steps(self):
        if self.py_env is None and self.tf_env is None:
            self.make_env()
        
        timestep = self.tf_env.reset()

        # TODO: does not support color generating adversary yet
        adv_iters = 50
        if self.colors is not None:
            adv_iters = 47
            for c in self.colors:
                timestep = self.tf_env.step_adversary(c)
                if self.video_fp is not None:
                    self.py_env.capture_frame()

        # TODO: process sequence or process agent
        if self.mode == "sequence":
            for s in self.sequence:
                _, _, done, _ = self.tf_env.step_adversary(s)
                if self.video_fp is not None:
                    self.py_env.capture_frame()

        elif self.mode == "agent":
            # timestep = self.tf_env.reset()
            policy_state = self.agent.get_initial_state(1)  

            actions = []
            # steps = [timestep]

            for i in range(adv_iters):
                policy_step = self.agent.action(timestep, policy_state=policy_state)

                policy_state = policy_step.state
                actions.append(policy_step.action.numpy()[0])

                timestep = self.tf_env.step_adversary(policy_step.action)
                _, rew, disc, obs = timestep
                # steps.append(timestep)
                # rewards.append(rew.numpy()[0])

                # record video if relevant
                if hasattr(self.py_env, "capture_frame"):
                    self.py_env.capture_frame()

            print(actions)

        else:
            raise NotImplementedError

        return self.py_env, self.tf_env



def load_agent(checkpoint_path):
    agent = tf.compat.v2.saved_model.load(checkpoint_path)
    return agent


def plot_obs(obs: ts_lib.TimeStep):
    im = obs[3]['image'].numpy()[0].astype(np.float)
    im = np.transpose(im / im.max(), (1,0,2))
    plt.imshow(im)
    plt.show()
    return im


def run_agent_on_env(agent, tf_env, py_env):
    done = False
    rewards = []

    # make the environment
    timestep = tf_env.reset_agent()
    policy_state = agent.get_initial_state(1)

    actions = []
    steps = [timestep]
    
    #   while not done:
    for i in range(100):
        policy_step = agent.action(timestep, policy_state=policy_state)

        policy_state = policy_step.state
        actions.append(policy_step.action.numpy()[0])

        timestep = tf_env.step(policy_step.action)
        _, rew, disc, obs = timestep
        steps.append(timestep)

        rewards.append(rew.numpy()[0])

        # record video if relevant
        if hasattr(py_env, "capture_frame"):
            py_env.capture_frame()
        
    print(actions)
    print(rewards)
 
    return steps


def backwards_compatible_timestep(new_ts):
    """Remove new observations added in later versions of the environment."""
    old_obs = {"image": new_ts.observation["image"], "time_step": new_ts.observation["time_step"]}
    return ts_lib.TimeStep(new_ts.step_type, new_ts.reward, new_ts.discount, old_obs)


def main():
    # TODO
    # source_shift = []
    # target_shift = []
    
    # all_source_agents = ["/home/charlie/SDRIVE/datasets/no_shift/policy_saved_model/agent/0/policy_000499950"]
    # all_source_agents = ["/home/ddobre/Projects/mila/game_theory/dynamic_shift/policy_000499950"]
    # all_source_agents = ["/home/ddobre/Projects/mila/game_theory/dynamic_shift/new_runs/policy_000038700"]
    #all_source_agents = [
    #    "/home/ddobre/Projects/mila/game_theory/dynamic_shift/new_runs/policy_000329400"
    #]

    agents_savedmodel = "/home/ddobre/Projects/game_theory/saved_models/static_shift/protagonist/policy_000289800"
    agent = load_agent(agents_savedmodel)

    corridor = AdvEnv(sequence=[12, 0, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
                      colors=[5, 1, 6],  # wall, goal, floor: gray, green, black
                    #   video_fp="video.mp4"
                )

    cluttered10 = BaseEnv('MultiGrid-Cluttered10-Minigrid-v0')
    cluttered40 = BaseEnv('MultiGrid-Cluttered40-Minigrid-v0')

    pyenv, tfenv = corridor.make_env()
    steps = run_agent_on_env(agent, tfenv, pyenv)
